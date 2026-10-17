"""Export YOLO-World with a fixed vocabulary as a Raspberry Pi model folder (no training, no CLIP on the Pi).

    python -m scripts.export_world_pi --classes keys wallet phone --imgsz 320 --calib frames/ \
        --out lastscene-data/models

set_classes() runs the CLIP text encoder once, here on the laptop, and stores the class embeddings in the
detection head. The exported ONNX is then a plain detector with a [1, 4 + nc, N] output: the same contract
as a trained model, so lastscene_pipeline.runtime.OnnxYoloDetector, bench and detect load it unchanged.
Class names go in classes.json (they are the dashboard labels). Prompts can differ: "keys" is prompted as
"bunch of keys" (runtime.PROMPT_HINTS, measured to work much better).

INT8 uses the same recipe as lastscene_pipeline.export: static QDQ, per-channel symmetric, head kept FP32,
calibrated on --calib folders (use real frames from the device's own camera).
"""

import argparse
import datetime
import importlib.metadata
import json
import shutil
from pathlib import Path

import cv2

from lastscene_pipeline.export import agreement, calibration_images, head_nodes
from lastscene_pipeline.runtime import PROMPT_HINTS, letterbox, sha256


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--classes", nargs="+", required=True, help="dashboard class names, in class-id order")
    p.add_argument("--base", default="yolov8s-worldv2.pt")
    p.add_argument("--imgsz", type=int, default=320)
    p.add_argument("--calib", action="append", required=True, help="folder of real camera frames (repeatable)")
    p.add_argument("--out", type=Path, default=Path("lastscene-data/models"))
    p.add_argument("--version", help="folder name; default world-<base>-<imgsz>-<utc time>")
    args = p.parse_args()

    import onnx
    from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
    from onnxruntime.quantization.shape_inference import quant_pre_process
    from ultralytics import YOLOWorld

    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    version = args.version or f"world-{Path(args.base).stem}-{args.imgsz}-{stamp}"
    target = args.out / version
    target.mkdir(parents=True, exist_ok=False)
    classes = list(args.classes)
    prompts = [PROMPT_HINTS.get(c, c) for c in classes]

    model = YOLOWorld(args.base)
    model.set_classes(prompts)
    exported = model.export(format="onnx", imgsz=args.imgsz, opset=17, simplify=True, dynamic=False,
                            batch=1, device="cpu")
    fp32 = target / "model-fp32.onnx"
    shutil.move(str(exported), fp32)
    onnx.checker.check_model(str(fp32))

    images = calibration_images(args.calib)
    if not images:
        raise SystemExit("no calibration images in --calib")
    pre = target / "model-pre.onnx"
    quant_pre_process(str(fp32), str(pre), skip_optimization=True)
    graph = onnx.load(str(pre))
    input_name = graph.graph.input[0].name
    keep_fp32 = head_nodes(graph)

    class Reader(CalibrationDataReader):
        def __init__(self):
            self.files = iter(images)

        def get_next(self):
            for f in self.files:
                img = cv2.imread(str(f))
                if img is not None:
                    return {input_name: letterbox(img, args.imgsz)[0]}
            return None

    int8 = target / "model-int8.onnx"
    quantize_static(str(pre), str(int8), Reader(), quant_format=QuantFormat.QDQ,
                    activation_type=QuantType.QInt8, weight_type=QuantType.QInt8, per_channel=True,
                    op_types_to_quantize=["Conv", "MatMul"], nodes_to_exclude=keep_fp32,
                    extra_options={"ActivationSymmetric": True, "WeightSymmetric": True})
    pre.unlink(missing_ok=True)
    check = agreement(fp32, int8, images[:60], args.imgsz, classes, confidence=.05)

    (target / "classes.json").write_text(json.dumps(
        {"version": version, "classes": classes, "items": [], "prompts": prompts}, indent=2), encoding="utf-8")
    (target / "metrics.json").write_text(json.dumps(
        {"version": version, "base_model": Path(args.base).name, "imgsz": args.imgsz, "epochs": 0,
         "note": "not trained: YOLO-World set_classes vocabulary baked into the export"}, indent=2),
        encoding="utf-8")
    manifest = {
        "version": version, "base_model": Path(args.base).name, "classes": classes, "prompts": prompts,
        "imgsz": args.imgsz, "input": [1, 3, args.imgsz, args.imgsz],
        "output": f"[1, {4 + len(classes)}, N] raw cx,cy,w,h + class scores",
        "preprocessing": "BGR->RGB, centered letterbox pad 114, NCHW float32 /255",
        "postprocessing": "max class score, per-class NMS IoU 0.7, drop boxes over half the frame",
        "files": {f.name: {"sha256": sha256(f), "bytes": f.stat().st_size} for f in (fp32, int8)},
        "int8": {"calibration_images": len(images), "head_kept_fp32": len(keep_fp32), "agreement_with_fp32": check},
        "exported_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "libraries": {lib: importlib.metadata.version(lib) for lib in ("ultralytics", "onnx", "onnxruntime")},
    }
    (target / "model.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (args.out / "current").write_text(version, encoding="utf-8")
    print(f"{target}: INT8 {int8.stat().st_size / 1e6:.1f} MB, FP32 {fp32.stat().st_size / 1e6:.1f} MB, "
          f"agreement {check}")


if __name__ == "__main__":
    main()
