"""Reproducible experimental FP32 + static QDQ INT8 export using development only."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil

import cv2
from memory_engine.enrollment import atomic_json
from memory_engine.onnx_detector import CLASSES, prepare


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", required=True)
    p.add_argument("--clips", required=True)
    p.add_argument("--out", default="data/onnx")
    p.add_argument("--sizes", nargs="+", type=int, default=[640, 320])
    p.add_argument("--split", default="eval_manifests/enrollment_sessions.json")
    args = p.parse_args()
    from ultralytics import YOLO
    from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
    from onnxruntime.quantization.shape_inference import quant_pre_process
    import onnxruntime as ort
    import onnx
    split = json.loads(Path(args.split).read_text())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    source = out / "source.pt"
    if not source.exists():
        shutil.copyfile(args.weights, source)
    if sha(source) != sha(args.weights):
        raise ValueError("existing source checkpoint differs")
    calibration = []
    for clip in split["development"]:
        path = Path(args.clips) / clip
        cap = cv2.VideoCapture(str(path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps:
            raise ValueError(f"missing/unreadable development recording {clip}")
        duration = cap.get(cv2.CAP_PROP_FRAME_COUNT)/fps*1000
        for ms in range(0, int(duration), 1000):
            cap.set(cv2.CAP_PROP_POS_MSEC, ms)
            ok, frame = cap.read()
            if ok:
                calibration.append((clip, cap.get(cv2.CAP_PROP_POS_MSEC), frame))
        cap.release()
    for size in args.sizes:
        fp = out / f"keys-s{size}-fp32.onnx"
        if not fp.exists():
            model = YOLO(str(source))
            model.set_classes(CLASSES)
            exported = model.export(format="onnx", imgsz=size, dynamic=False, simplify=False,
                                    opset=17, device="cpu", batch=1)
            shutil.move(str(exported), fp)
        onnx.checker.check_model(str(fp))
        pre = out / f"keys-s{size}-pre.onnx"
        quant_pre_process(str(fp), str(pre), skip_optimization=True)
        session = ort.InferenceSession(str(pre), providers=["CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name
        del session
        class Reader(CalibrationDataReader):
            def __init__(self):
                self.frames = iter(calibration)
            def get_next(self):
                entry = next(self.frames, None)
                return {input_name: prepare(entry[2], size)[0]} if entry else None
        quant = out / f"keys-s{size}-int8.onnx"
        quantize_static(str(pre), str(quant), Reader(), quant_format=QuantFormat.QDQ,
                        activation_type=QuantType.QInt8, weight_type=QuantType.QInt8,
                        per_channel=True, op_types_to_quantize=["Conv", "MatMul"],
                        extra_options={"ActivationSymmetric": True, "WeightSymmetric": True})
        for artifact in [fp, quant]:
            atomic_json(artifact.with_suffix(".json"), dict(classes=CLASSES, size=size,
                sha256=sha(artifact), source_sha256=sha(source), split_sha256=sha(args.split),
                status="experimental; host only; release gates pending", bytes=artifact.stat().st_size,
                calibration=[dict(clip=c, frame_ms=t) for c,t,_ in calibration] if artifact == quant else [],
                libraries={lib: importlib.metadata.version(lib) for lib in ["torch", "ultralytics", "onnx", "onnxruntime"]},
                preprocessing="BGR to RGB, centered square letterbox114, NCHW float32 /255",
                postprocessing="raw xywh, max class, class-agnostic NMS IoU .7 before distractor suppression"))
        print(f"Exported {fp} and {quant}", flush=True)


if __name__ == "__main__":
    main()
