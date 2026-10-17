"""train: fine-tune a detector on the synced dataset and publish a model detect can load.

    python -m lastscene_pipeline.train [--data DIR/data.yaml] [--base MODEL] [--epochs N] [--imgsz PX] [--out DIR]

Output: models/<version>/ with best.pt, classes.json (class order unchanged), metrics.json, and for the
Raspberry Pi model-int8.onnx + model-fp32.onnx + model.json (skip with --no-export). models/current (a
one-line file naming the folder) moves only after a successful run, and only if the new model's map50 is
not lower than the current one's on the same photos.

Base: yolo11s.pt by default, chosen for the Pi: about 10 MB as INT8, where YOLO-World s is about 13 MB.
Dylan's spec defaults to yolov8s-worldv2.pt; any Ultralytics detect checkpoint works via --base.
A closed-vocabulary model can't detect an item from its name alone: items with fewer than 5 photos stay in
classes.json, but the trained model will likely miss them (on the laptop, detect without a model uses
YOLO-World set_classes instead).
map50 is measured on the training photos (data.yaml's val = train, per the spec), so it is optimistic.

Exit codes: 0 done (also when the old model is kept), 1 bad input, 3 no photos.
"""

import argparse
import datetime
import json
import os
import shutil
from pathlib import Path

from lastscene_pipeline import EXIT_CONFIG, EXIT_NOTHING, EXIT_OK
from lastscene_pipeline.common import PipelineError, data_dir, read_json, resolve_model_dir, run_main

DEFAULT_BASE = "yolo11s.pt"
MIN_PHOTOS = 5


def photos_per_class(dataset: Path, n_classes: int) -> list[int]:
    counts = [0] * n_classes
    for f in (dataset / "labels" / "train").glob("*.txt"):
        for line in f.read_text(encoding="utf-8").split("\n"):
            if line.strip():
                counts[int(line.split()[0])] += 1
    return counts


def set_current(out: Path, version: str):
    """Point models/current at models/<version> atomically (a one-line file; works without symlink rights)."""
    current = out / "current"
    if current.is_symlink() or current.is_dir():
        current.unlink() if current.is_symlink() else shutil.rmtree(current)
    tmp = out / ".current.tmp"
    tmp.write_text(version + "\n", encoding="utf-8")
    os.replace(tmp, current)


def validate(weights, data_yaml, imgsz, device) -> tuple[float, dict]:
    from ultralytics import YOLO

    v = YOLO(str(weights)).val(data=str(data_yaml), imgsz=imgsz, device=device, plots=False, verbose=False,
                               workers=0, batch=8)
    per = {int(c): float(ap) for c, ap in zip(v.box.ap_class_index, v.box.ap50)}
    return float(v.box.map50), per


def run_train(data_yaml: Path, base: str, epochs: int, imgsz: int, out: Path, export_pi: bool = True,
              calib=(), device=None, log=print) -> int:
    dataset = data_yaml.parent
    for needed in (data_yaml, dataset / "version", dataset / "classes.json"):
        if not needed.exists():
            raise PipelineError(EXIT_CONFIG, f"{needed} is missing; run sync first")
    version = (dataset / "version").read_text(encoding="utf-8").strip()
    vocab = read_json(dataset / "classes.json")
    classes = vocab["classes"]
    counts = photos_per_class(dataset, len(classes))
    if not sum(counts):
        log("no photos in the dataset; nothing to train (models/current unchanged)")
        return EXIT_NOTHING
    for name, n in zip(classes, counts):
        if n < MIN_PHOTOS:
            log(f"'{name}' has {n} photo(s) (< {MIN_PHOTOS}): it stays in classes.json, but this model will likely "
                f"miss it; add photos on the Teach items page")
    from ultralytics import YOLO  # after the checks: they work without the laptop libraries (Pi tests)

    if device is None:
        import torch

        device = "0" if torch.cuda.is_available() else "cpu"

    runs = out / ".runs"
    model = YOLO(base)
    # patience: with a handful of photos the score sits near 0 for the first epochs; a short patience
    # stopped a 14-photo run at epoch 11 with map50 0.02
    model.train(data=str(data_yaml), epochs=epochs, imgsz=imgsz, patience=max(25, epochs // 2), device=device,
                project=str(runs.resolve()), name=version, exist_ok=True, plots=False, verbose=False, seed=0,
                deterministic=True, workers=0, batch=16)
    trained_epochs = model.trainer.epoch + 1
    best = runs / version / "weights" / "best.pt"
    if not best.exists():
        raise PipelineError(EXIT_CONFIG, f"training produced no {best}")
    map50, per = validate(best, data_yaml, imgsz, device)

    target = out / version
    target.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(best, target / "best.pt")
    shutil.copyfile(dataset / "classes.json", target / "classes.json")
    metrics = {
        "version": version, "base_model": Path(base).name, "epochs": trained_epochs, "imgsz": imgsz,
        "trained_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "map50": round(map50, 4),
        "per_class": {name: {"map50": round(per[i], 4) if i in per else None, "photos": counts[i]}
                      for i, name in enumerate(classes)},
        "map50_on": "training photos (data.yaml val = train)", "dataset": str(dataset.resolve()),
    }
    (target / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    log(f"trained {version}: {trained_epochs} epochs, map50 {map50:.3f} on the training photos -> {target}")

    if export_pi:
        from lastscene_pipeline.export import export

        try:
            export(target, calib, log=log)
        except Exception as exc:  # noqa: BLE001  the laptop can still use best.pt; say so and keep going
            log(f"warning: INT8 export for the Pi failed ({type(exc).__name__}: {exc}); best.pt is still usable")

    current = resolve_model_dir(out / "current")
    if current is not None and current.resolve() != target.resolve() and (current / "best.pt").exists():
        same_classes = read_json(current / "classes.json").get("classes") == classes
        if same_classes:
            old_map50, _ = validate(current / "best.pt", data_yaml, imgsz, device)
            if old_map50 > map50:
                log(f"warning: new map50 {map50:.3f} < current {old_map50:.3f} on the same photos; "
                    f"keeping models/current = {current.name}")
                return EXIT_OK
    set_current(out, version)
    log(f"models/current -> {version}")
    return EXIT_OK


def main():
    root = data_dir()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=root / "dataset" / "data.yaml")
    p.add_argument("--base", default=DEFAULT_BASE, help="any Ultralytics detect checkpoint (spec default: yolov8s-worldv2.pt)")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--imgsz", type=int, default=640, help="training and inference size must match (Pi: try 416 or 320)")
    p.add_argument("--out", type=Path, default=root / "models")
    p.add_argument("--no-export", action="store_true", help="skip the INT8 ONNX export for the Pi")
    p.add_argument("--calib", action="append", default=[], help="extra folder of real camera frames for INT8 calibration")
    p.add_argument("--device", help="e.g. 0 or cpu (default: GPU if available)")
    args = p.parse_args()
    return run_train(args.data, args.base, args.epochs, args.imgsz, args.out, not args.no_export, args.calib, args.device)


if __name__ == "__main__":
    run_main(main)
