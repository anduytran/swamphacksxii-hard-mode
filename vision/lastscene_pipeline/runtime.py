"""Detectors detect can load, all returning memory_engine Detection(label=<dashboard class name>, conf, bbox px).

    OnnxYoloDetector    model-int8.onnx (or model-fp32.onnx) + model.json from train/export: the Raspberry Pi path.
                        Needs only onnxruntime, OpenCV and numpy.
    UltralyticsDetector best.pt from train: the laptop path (GPU if available).
    world_fallback      no trained model: YOLO-World set_classes(class names). Laptop only: the CLIP text
                        encoder it needs is ~350 MB, too big for a Pi Zero 2 W.
"""

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from memory_engine.detector import Detection, drop_oversized

# Open-vocabulary prompts that work better than the bare class name (measured: "bunch of keys" >> "keys").
PROMPT_HINTS = {"keys": "bunch of keys", "key": "bunch of keys", "car keys": "bunch of keys"}
MAX_AREA_FRAC = 0.5  # a handheld item never fills half the frame; such boxes are noise (see detector.py)


def letterbox(image, size):
    """Ultralytics-style centered letterbox to size x size, pad 114. Returns NCHW float32 0..1 and (gain, left, top)."""
    h, w = image.shape[:2]
    gain = min(size / h, size / w)
    width, height = round(w * gain), round(h * gain)
    left, top = round((size - width) / 2 - .1), round((size - height) / 2 - .1)
    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
    padded = cv2.copyMakeBorder(resized, top, size - height - top, left, size - width - left,
                                cv2.BORDER_CONSTANT, value=(114, 114, 114))
    tensor = np.ascontiguousarray(padded[:, :, ::-1].transpose(2, 0, 1)[None], dtype=np.float32) / 255
    return tensor, (gain, left, top)


def decode(output, transform, shape, classes, confidence=.25, iou=.7):
    """Raw YOLOv8/YOLO11 head output [1, 4 + nc, N] (cx, cy, w, h in input pixels, then class scores)."""
    nc = len(classes)
    if output.ndim != 3 or output.shape[0] != 1 or output.shape[1] != 4 + nc:
        raise ValueError(f"expected raw YOLO output [1, {4 + nc}, N], got {list(output.shape)}")
    rows = output[0].T
    cls = rows[:, 4:].argmax(axis=1)
    scores = rows[np.arange(len(rows)), cls + 4]
    keep = np.isfinite(rows).all(axis=1) & (scores >= confidence)
    rows, cls, scores = rows[keep], cls[keep], scores[keep]
    if not len(rows):
        return []
    boxes = np.column_stack((rows[:, 0] - rows[:, 2] / 2, rows[:, 1] - rows[:, 3] / 2, rows[:, 2:4]))
    offset = boxes.copy()  # per-class NMS: shift each class far apart, then one NMS call
    offset[:, :2] += cls[:, None] * 10000.0
    indices = cv2.dnn.NMSBoxes(offset.tolist(), scores.tolist(), confidence, iou)
    gain, left, top = transform
    h, w = shape[:2]
    result = []
    for i in np.asarray(indices).reshape(-1):
        x, y, bw, bh = boxes[i]
        box = (int(np.clip((x - left) / gain, 0, w)), int(np.clip((y - top) / gain, 0, h)),
               int(np.clip((x + bw - left) / gain, 0, w)), int(np.clip((y + bh - top) / gain, 0, h)))
        if box[2] > box[0] and box[3] > box[1]:
            result.append(Detection(classes[cls[i]], round(float(scores[i]), 3), box))
    return drop_oversized(result, shape, MAX_AREA_FRAC)


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class OnnxYoloDetector:
    def __init__(self, model, confidence=.25, threads=4, verify_hash=True, low_memory=False):
        import onnxruntime as ort

        path = Path(model)
        self.manifest = json.loads(path.with_name("model.json").read_text(encoding="utf-8"))
        entry = self.manifest["files"].get(path.name)
        if entry is None:
            raise ValueError(f"{path.name} is not listed in model.json")
        if verify_hash and entry["sha256"] != sha256(path):
            raise ValueError(f"{path.name}: hash differs from model.json (partial copy?)")
        self.classes = self.manifest["classes"]
        self.size = self.manifest["imgsz"]
        self.confidence = confidence
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        if low_memory:  # Pi Zero 2 W has 512 MB: no pre-grown arena / cached allocation plans
            options.enable_cpu_mem_arena = False
            options.enable_mem_pattern = False
            # No spin-waiting between ops: on the Zero 2 W (Oct 17, 2026, Pin model, 4 threads) 0.64 s instead
            # of 0.615 s per frame, but 1.9 instead of 2.5 CPU-seconds, and with speech-to-text running in the
            # same process the detector took 1.13 s per frame instead of 1.54 s.
            options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        self.session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
        self.input = self.session.get_inputs()[0]
        if list(self.input.shape) != [1, 3, self.size, self.size]:
            raise ValueError(f"model input {self.input.shape} != [1, 3, {self.size}, {self.size}]")
        self.name = f"{self.manifest['base_model']}@{self.manifest['version']} {path.stem} {self.size}px"

    def detect(self, image):
        tensor, transform = letterbox(image, self.size)
        output = self.session.run(None, {self.input.name: tensor})[0]
        return decode(output, transform, image.shape, self.classes, self.confidence)


class UltralyticsDetector:
    def __init__(self, weights, classes, imgsz, confidence=.25):
        from ultralytics import YOLO

        self.model = YOLO(str(weights))
        self.classes, self.imgsz, self.confidence = list(classes), imgsz, confidence
        names = [self.model.names[i] for i in sorted(self.model.names)]
        if names != self.classes:
            raise ValueError(f"{weights}: model classes {names} != classes.json {self.classes}")
        self.name = f"{Path(weights).parent.name} best.pt {imgsz}px"

    def detect(self, image):
        r = self.model.predict(image, conf=self.confidence, imgsz=self.imgsz, verbose=False)[0]
        dets = [Detection(self.classes[int(b.cls)], round(float(b.conf), 3), tuple(int(v) for v in b.xyxy[0].tolist()))
                for b in r.boxes]
        return drop_oversized(dets, image.shape, MAX_AREA_FRAC)


class WorldFallback:
    """YOLO-World with set_classes(class names): detect works before any training (spec: training is optional)."""

    def __init__(self, classes, confidence=.05, base="yolov8m-worldv2.pt", imgsz=960):
        from memory_engine.detector import YoloWorldDetector
        from memory_engine.presets import DISTRACTORS

        self.classes = list(classes)
        self.prompt_to_class = {PROMPT_HINTS.get(c, c): c for c in self.classes}
        self.inner = YoloWorldDetector(list(self.prompt_to_class), model=base, conf=confidence, imgsz=imgsz,
                                       half=_cuda(), distractors=DISTRACTORS)
        self.name = f"yolo-world {Path(base).stem} {imgsz}px set_classes (no trained model)"

    def detect(self, image):
        return [Detection(self.prompt_to_class[d.label], d.conf, d.bbox) for d in self.inner.detect(image)]


def _cuda() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


def _has(module) -> bool:
    import importlib.util

    return importlib.util.find_spec(module) is not None


def small_ram(limit_mb: int = 1024) -> bool:
    """True on boards like the Pi Zero 2 W (512 MB). Linux only; elsewhere False."""
    import os

    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") < limit_mb * 1024 * 1024
    except (AttributeError, ValueError, OSError):
        return False


def load_detector(model_dir: Path | None, classes: list[str], confidence: float, runtime: str = "auto",
                  threads: int = 4, world_base: str = "yolov8m-worldv2.pt", low_memory: bool | None = None):
    """runtime: auto (best.pt with Ultralytics if installed, else INT8 ONNX), onnx, onnx-fp32, torch.
    low_memory (ONNX only): None = on when the machine has < 1 GB RAM (no memory arena, no spin-waiting)."""
    low_memory = small_ram() if low_memory is None else low_memory
    if model_dir is not None:
        best, int8, fp32 = model_dir / "best.pt", model_dir / "model-int8.onnx", model_dir / "model-fp32.onnx"
        imgsz = json.loads((model_dir / "metrics.json").read_text(encoding="utf-8"))["imgsz"] \
            if (model_dir / "metrics.json").exists() else 640
        if runtime in ("auto", "torch") and best.exists() and _has("ultralytics"):
            return UltralyticsDetector(best, classes, imgsz, confidence)
        if runtime in ("auto", "onnx") and int8.exists():
            return _checked(OnnxYoloDetector(int8, confidence, threads, low_memory=low_memory), classes)
        if runtime == "onnx-fp32" and fp32.exists():
            return _checked(OnnxYoloDetector(fp32, confidence, threads, low_memory=low_memory), classes)
        raise FileNotFoundError(f"{model_dir}: no usable model for runtime={runtime} "
                                f"(best.pt needs ultralytics; ONNX needs model-int8.onnx + model.json from train)")
    if not _has("ultralytics"):
        raise FileNotFoundError("no trained model, and YOLO-World (set_classes) needs the laptop environment; "
                                "run train on the laptop and copy models/<version>/ here")
    return WorldFallback(classes, min(confidence, 0.05), base=world_base)


def _checked(det, classes):
    if det.classes != list(classes):
        raise ValueError(f"model classes {det.classes} != classes.json {list(classes)}")
    return det
