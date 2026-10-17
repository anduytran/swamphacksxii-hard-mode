"""Export a trained model for the Raspberry Pi: FP32 ONNX + static INT8 ONNX + model.json.

    python -m lastscene_pipeline.export --model lastscene-data/models/current [--calib DIR ...]

INT8: static QDQ, per-channel symmetric weights, calibrated on the dataset photos (plus any --calib
folders of real camera frames, which represent the wearable better). The detection head's box decoding
(DFL, anchors, final Concat) stays FP32: boxes in pixels and class scores 0..1 share one output tensor,
and quantizing them together costs accuracy for almost no size. model.json records hashes, classes,
input size, and how closely INT8 agrees with FP32 on the calibration photos (not a held-out accuracy test).
Runs on the laptop (needs ultralytics, onnx, onnxruntime); the Pi only loads the result.
"""

import argparse
import datetime
import importlib.metadata
import json
import shutil
from pathlib import Path

import cv2
import numpy as np

from lastscene_pipeline import EXIT_CONFIG, EXIT_OK
from lastscene_pipeline.common import PipelineError, read_json, resolve_model_dir, run_main
from lastscene_pipeline.runtime import decode, letterbox, sha256

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def calibration_images(folders, limit=300):
    files = sorted(f for d in folders for f in Path(d).rglob("*") if f.suffix.lower() in IMAGE_SUFFIXES)
    step = max(1, len(files) // limit)
    return files[::step][:limit]


def head_nodes(model) -> list[str]:
    """Detect-head convs kept in FP32: the DFL conv and the last box (cv2) / class (cv3) conv per scale.
    Only Conv/MatMul are quantized, so everything after them (decoding, Concat) is FP32 anyway."""
    import re

    layers = [int(m.group(1)) for n in model.graph.node if (m := re.match(r"/model\.(\d+)/", n.name))]
    head = f"/model.{max(layers)}/"
    last_conv = re.compile(r"/cv[23]\.\d+/cv[23]\.\d+\.2/Conv")
    return [n.name for n in model.graph.node if n.op_type == "Conv" and n.name.startswith(head)
            and ("dfl" in n.name or last_conv.search(n.name))]


def agreement(fp32, int8, images, size, classes, confidence=.25) -> dict:
    """How INT8 detections match FP32 on the same images: matched share (IoU >= .5, same class), score drift."""
    import onnxruntime as ort

    sessions = [ort.InferenceSession(str(p), providers=["CPUExecutionProvider"]) for p in (fp32, int8)]
    name = sessions[0].get_inputs()[0].name
    ref_total = matched = 0
    drifts = []
    for f in images:
        img = cv2.imread(str(f))
        if img is None:
            continue
        tensor, tf = letterbox(img, size)
        ref, got = (decode(s.run(None, {name: tensor})[0], tf, img.shape, classes, confidence) for s in sessions)
        ref_total += len(ref)
        for r in ref:
            best = max((g for g in got if g.label == r.label), key=lambda g: _iou(g.bbox, r.bbox), default=None)
            if best is not None and _iou(best.bbox, r.bbox) >= .5:
                matched += 1
                drifts.append(best.conf - r.conf)
    return {"images": len(images), "fp32_detections": ref_total,
            "int8_matched": round(matched / ref_total, 3) if ref_total else None,
            "mean_score_change": round(float(np.mean(drifts)), 4) if drifts else None,
            "note": "on calibration photos; not a held-out accuracy test"}


def _iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 0.0


def export(model_dir: Path, calib_folders=(), log=print) -> dict:
    import onnx
    from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
    from onnxruntime.quantization.shape_inference import quant_pre_process
    from ultralytics import YOLO

    metrics = read_json(model_dir / "metrics.json")
    classes = read_json(model_dir / "classes.json")["classes"]
    size = metrics["imgsz"]
    dataset = Path(metrics["dataset"])
    images = calibration_images([dataset / "images" / "train", *calib_folders])
    if not images:
        raise PipelineError(EXIT_CONFIG, "no calibration images (dataset photos or --calib folders)")

    fp32 = model_dir / "model-fp32.onnx"
    exported = YOLO(str(model_dir / "best.pt")).export(format="onnx", imgsz=size, opset=17, simplify=True,
                                                       dynamic=False, batch=1, device="cpu")
    shutil.move(str(exported), fp32)
    onnx.checker.check_model(str(fp32))
    pre = model_dir / "model-pre.onnx"
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
                    return {input_name: letterbox(img, size)[0]}
            return None

    int8 = model_dir / "model-int8.onnx"
    quantize_static(str(pre), str(int8), Reader(), quant_format=QuantFormat.QDQ,
                    activation_type=QuantType.QInt8, weight_type=QuantType.QInt8, per_channel=True,
                    op_types_to_quantize=["Conv", "MatMul"], nodes_to_exclude=keep_fp32,
                    extra_options={"ActivationSymmetric": True, "WeightSymmetric": True})
    pre.unlink(missing_ok=True)
    check = agreement(fp32, int8, images[:60], size, classes)
    manifest = {
        "version": metrics["version"], "base_model": metrics["base_model"], "classes": classes, "imgsz": size,
        "input": [1, 3, size, size], "output": f"[1, {4 + len(classes)}, N] raw cx,cy,w,h + class scores",
        "preprocessing": "BGR->RGB, centered letterbox pad 114, NCHW float32 /255",
        "postprocessing": "max class score, per-class NMS IoU 0.7, drop boxes over half the frame",
        "files": {p.name: {"sha256": sha256(p), "bytes": p.stat().st_size} for p in (fp32, int8)},
        "int8": {"calibration_images": len(images), "head_kept_fp32": len(keep_fp32), "agreement_with_fp32": check},
        "exported_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "libraries": {lib: importlib.metadata.version(lib) for lib in ("ultralytics", "onnx", "onnxruntime")},
    }
    (model_dir / "model.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    mb = int8.stat().st_size / 1e6
    log(f"exported {int8} ({mb:.1f} MB INT8, {fp32.stat().st_size / 1e6:.1f} MB FP32); "
        f"INT8 matches {check['int8_matched']} of FP32 detections on {check['images']} calibration photos")
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", type=Path, required=True, help="models/<version> or models/current")
    p.add_argument("--calib", action="append", default=[], help="extra folder of real camera frames (repeatable)")
    args = p.parse_args()
    model_dir = resolve_model_dir(args.model)
    if model_dir is None or not (model_dir / "best.pt").exists():
        raise PipelineError(EXIT_CONFIG, f"{args.model}: no trained model (best.pt)")
    export(model_dir, args.calib)
    return EXIT_OK


if __name__ == "__main__":
    run_main(main)
