"""Local host enrollment. References personalize matching, never network weights.

Single-writer store; stop the bridge before updating references. Identity results
are experimental and similarity is not detector confidence or a probability.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import uuid

import cv2
import numpy as np


def unit(vector):
    vector = np.asarray(vector, dtype=np.float32)
    if vector.ndim != 1 or not np.isfinite(vector).all() or np.linalg.norm(vector) <= 0:
        raise ValueError("embedding must be a finite nonzero vector")
    return vector / np.linalg.norm(vector)


def crop_image(image, box):
    if image is None or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("expected a BGR image")
    if len(box) != 4 or any(not isinstance(v, (int, np.integer)) for v in box):
        raise ValueError("crop bounds must be four integers")
    x1, y1, x2, y2 = box
    h, w = image.shape[:2]
    if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
        raise ValueError("crop is empty or outside image bounds")
    return image[y1:y2, x1:x2].copy()


def atomic_json(path, data):
    path = Path(path)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        temporary.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class ClipEncoder:
    def __init__(self, checkpoint, device=None):
        import clip
        import torch
        path = Path(checkpoint).resolve(strict=True)
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.preprocess = clip.load(str(path), device=self.device, jit=False)
        self.model.float().eval()
        self.version = {"sha256": digest, "preprocessing": "openai-clip-vit-b32-rgb-bicubic-center224-v1"}

    def encode(self, crop):
        import torch
        from PIL import Image
        image = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        with torch.inference_mode():
            vector = self.model.encode_image(self.preprocess(image).unsqueeze(0).to(self.device))
        return unit(vector[0].float().cpu().numpy())


class EnrollmentStore:
    def __init__(self, root, encoder=None):
        self.root, self.encoder = Path(root), encoder
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, item_id):
        return self.root / str(uuid.UUID(item_id))

    def items(self):
        return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(self.root.glob("*/item.json"))]

    def register(self, display_name, category_prompt):
        name, category = display_name.strip(), category_prompt.strip()
        if not name or not category or any(c in name for c in "/\\\n\r"):
            raise ValueError("nonempty name/category required; names cannot contain slashes or newlines")
        from lastscene_bridge.observation import item_label
        label = item_label(name)
        if any(i["label"] == label for i in self.items()):
            raise ValueError("an item already uses this backend label")
        item_id = str(uuid.uuid4())
        directory = self._path(item_id)
        directory.mkdir()
        atomic_json(directory / "item.json", dict(item_id=item_id, display_name=name, label=label,
                    category_prompt=category, enrollment_version=0, examples=[], encoder_version=None))
        return item_id

    def add_example(self, item_id, image, box, *, source, frame_ms, confirmed=False):
        if confirmed is not True:
            raise ValueError("explicit crop confirmation required")
        if not source or not np.isfinite(frame_ms) or frame_ms < 0:
            raise ValueError("source and nonnegative frame position required")
        crop = crop_image(image, box)
        if self.encoder is None:
            raise ValueError("encoder required")
        directory = self._path(item_id)
        item = json.loads((directory / "item.json").read_text(encoding="utf-8"))
        if item["encoder_version"] not in (None, self.encoder.version):
            raise ValueError("encoder version mismatch; rebuild references")
        vector = unit(self.encoder.encode(crop))
        if item["examples"] and len(vector) != len(item["examples"][0]["embedding"]):
            raise ValueError("embedding dimension mismatch")
        example_id = str(uuid.uuid4())
        filename = example_id + ".png"
        if not cv2.imwrite(str(directory / filename), crop):
            raise ValueError("could not save reference crop")
        item["encoder_version"] = self.encoder.version
        item["enrollment_version"] += 1
        item["examples"].append(dict(example_id=example_id, crop=filename, box=list(box),
                                     source=str(source), frame_ms=float(frame_ms), confirmed=True,
                                     embedding=vector.tolist()))
        try:
            atomic_json(directory / "item.json", item)
        except Exception:
            (directory / filename).unlink(missing_ok=True)
            raise
        return example_id

    def delete(self, item_id):
        shutil.rmtree(self._path(item_id))

    def match(self, vector, category, threshold, margin):
        if not np.isfinite([threshold, margin]).all() or not -1 <= threshold <= 1 or not 0 <= margin <= 2:
            raise ValueError("invalid similarity threshold/margin")
        vector = unit(vector)
        scores = []
        for item in self.items():
            if item["category_prompt"] != category or not item["examples"]:
                continue
            if self.encoder is None or item["encoder_version"] != self.encoder.version:
                raise ValueError("encoder version mismatch; rebuild references")
            refs = np.asarray([e["embedding"] for e in item["examples"]], np.float32)
            if refs.shape[1] != len(vector) or not np.isfinite(refs).all():
                raise ValueError("invalid reference embeddings")
            score = float(np.max(refs @ vector))
            scores.append((score, item))
        scores.sort(key=lambda pair: pair[0], reverse=True)
        result = dict(status="unknown", item_id=None, label=None,
                      score=scores[0][0] if scores else None)
        if not scores or scores[0][0] < threshold:
            return result
        if len(scores) > 1 and scores[0][0] - scores[1][0] < margin:
            return {**result, "status": "ambiguous"}
        item = scores[0][1]
        return {**result, "status": "matched", "item_id": item["item_id"], "label": item["label"],
                "enrollment_version": item["enrollment_version"]}


class EnrolledDetector:
    def __init__(self, detector, store, threshold, margin, diagnostics=None):
        self.detector, self.store = detector, store
        self.threshold, self.margin, self.diagnostics = threshold, margin, diagnostics

    def detect(self, image):
        from memory_engine.detector import Detection
        accepted = []
        for detection in self.detector.detect(image):
            vector = self.store.encoder.encode(crop_image(image, detection.bbox))
            match = self.store.match(vector, detection.label, self.threshold, self.margin)
            if self.diagnostics:
                self.diagnostics({**match, "category": detection.label, "detector_confidence": detection.conf,
                                  "bbox": detection.bbox})
            if match["status"] == "matched":
                accepted.append(Detection(match["label"], detection.conf, detection.bbox))
        return accepted


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", default="data/enrollment")
    parser.add_argument("--checkpoint")
    sub = parser.add_subparsers(dest="command", required=True)
    register = sub.add_parser("register")
    register.add_argument("name")
    register.add_argument("--category", default="bunch of keys")
    sub.add_parser("list")
    delete = sub.add_parser("delete")
    delete.add_argument("item_id")
    add = sub.add_parser("add")
    add.add_argument("item_id")
    add.add_argument("--clip", required=True)
    add.add_argument("--frame-ms", type=float, required=True)
    add.add_argument("--box", type=int, nargs=4, required=True)
    add.add_argument("--confirmed", action="store_true")
    args = parser.parse_args()
    encoder = ClipEncoder(args.checkpoint) if args.command == "add" and args.checkpoint else None
    store = EnrollmentStore(args.store, encoder)
    if args.command == "register":
        print(store.register(args.name, args.category))
    elif args.command == "list":
        print(json.dumps([{k: v for k, v in i.items() if k != "examples"} |
                          {"example_count": len(i["examples"])} for i in store.items()], indent=2))
    elif args.command == "delete":
        store.delete(args.item_id)
    else:
        cap = cv2.VideoCapture(args.clip)
        try:
            cap.set(cv2.CAP_PROP_POS_MSEC, args.frame_ms)
            ok, frame = cap.read()
            if not ok:
                raise ValueError("cannot read requested frame")
            position = cap.get(cv2.CAP_PROP_POS_MSEC)
        finally:
            cap.release()
        print(store.add_example(args.item_id, frame, args.box, source=str(Path(args.clip).resolve()),
                                frame_ms=position, confirmed=args.confirmed))


if __name__ == "__main__":
    main()
