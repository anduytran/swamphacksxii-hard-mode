"""Experimental fixed-vocabulary ONNX adapter; CPU host measurements only."""
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from memory_engine.detector import Detection, drop_oversized

CLASSES = ["bunch of keys", "laptop keyboard", "laptop"]


def prepare(image, size):
    h, w = image.shape[:2]
    gain = min(size / h, size / w)
    width, height = round(w * gain), round(h * gain)
    left, top = round((size-width)/2-.1), round((size-height)/2-.1)
    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
    padded = cv2.copyMakeBorder(resized, top, size-height-top, left, size-width-left,
                               cv2.BORDER_CONSTANT, value=(114, 114, 114))
    tensor = np.ascontiguousarray(padded[:, :, ::-1].transpose(2, 0, 1)[None], dtype=np.float32)/255
    return tensor, (gain, left, top)


def decode(output, transform, shape, confidence=.05, iou=.7):
    if output.ndim != 3 or output.shape[0] != 1 or output.shape[1] != 7:
        raise ValueError("expected raw YOLO output [1,7,N] for the frozen three classes")
    rows = output[0].T
    classes = rows[:, 4:].argmax(axis=1)
    scores = rows[np.arange(len(rows)), classes+4]
    keep = np.isfinite(rows).all(axis=1) & (scores >= confidence)
    rows, classes, scores = rows[keep], classes[keep], scores[keep]
    boxes = np.column_stack((rows[:, 0]-rows[:, 2]/2, rows[:, 1]-rows[:, 3]/2, rows[:, 2:4]))
    indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), confidence, iou)
    gain, left, top = transform
    h, w = shape[:2]
    result = []
    for index in np.asarray(indices).reshape(-1):
        if classes[index] != 0:
            continue
        x, y, bw, bh = boxes[index]
        box = (int(np.clip((x-left)/gain, 0, w)), int(np.clip((y-top)/gain, 0, h)),
               int(np.clip((x+bw-left)/gain, 0, w)), int(np.clip((y+bh-top)/gain, 0, h)))
        if box[2] > box[0] and box[3] > box[1]:
            result.append(Detection(CLASSES[0], float(scores[index]), box))
    return drop_oversized(result, shape, .5)


class OnnxDetector:
    def __init__(self, model, confidence=.05, threads=4):
        import onnxruntime as ort
        path = Path(model)
        manifest = json.loads(path.with_suffix(".json").read_text())
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if manifest["classes"] != CLASSES or manifest["sha256"] != digest:
            raise ValueError("model hash/class map mismatch")
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        self.session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
        self.input = self.session.get_inputs()[0]
        self.size, self.confidence = manifest["size"], confidence
        if self.input.shape != [1, 3, self.size, self.size]:
            raise ValueError("incompatible model input")

    def detect(self, image):
        tensor, transform = prepare(image, self.size)
        output = self.session.run(None, {self.input.name: tensor})[0]
        return decode(output, transform, image.shape, self.confidence)
