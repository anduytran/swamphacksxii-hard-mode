"""Freeze a development-only threshold; evaluate held-out proposals after freeze.

Stages: freeze requires confirmed reference crops and reviewed development crops.
heldout writes proposals for visual review; summarize consumes that review without
changing the frozen threshold. No held-out scores enter threshold selection.
"""
import argparse
import json
from pathlib import Path
import time

import cv2
import numpy as np
from memory_engine.enrollment import ClipEncoder, EnrollmentStore, atomic_json, crop_image
from scripts.export_int8 import sha


def choose_threshold(positive, negative):
    if not positive or not negative or not np.isfinite(positive + negative).all():
        raise ValueError("reviewed development positives and negatives required")
    threshold = float(np.nextafter(np.float32(max(negative)), np.float32(np.inf)))
    if threshold > 1 or not any(v >= threshold for v in positive):
        raise ValueError("development scores cannot usefully separate; matching gate failed")
    return threshold


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["freeze", "heldout", "summarize"])
    p.add_argument("--review", default="data/enrollment_review/review.json")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--clips", required=True)
    p.add_argument("--weights", required=True)
    p.add_argument("--out", default="data/enrollment_experiment")
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    review = json.loads(Path(args.review).read_text())
    if args.stage == "summarize":
        rows = json.loads((out / "heldout.json").read_text())["candidates"]
        if any(r.get("review") not in ["key", "non_key"] or not r.get("reviewer") for r in rows):
            raise ValueError("all held-out proposals require visual review")
        summary = {}
        for name in review["split"]["held_out_positive"] + review["split"]["held_out_no_keys"]:
            clip = [r for r in rows if r["clip"] == name]
            keys = [r for r in clip if r["review"] == "key"]
            false = [r for r in clip if r["review"] == "non_key"]
            summary[name] = dict(proposals=len(clip), localized_key_proposals=len(keys),
                accepted_keys=sum(r["status"] == "matched" for r in keys),
                non_key_proposals=len(false), incorrect_matches=sum(r["status"] == "matched" for r in false),
                scores=[r["score"] for r in clip])
        atomic_json(out / "summary.json", summary)
        print(json.dumps(summary, indent=2))
        return
    encoder = ClipEncoder(args.checkpoint)
    store = EnrollmentStore(out / "store", encoder)
    if args.stage == "freeze":
        if (out / "frozen.json").exists() or store.items():
            raise ValueError("experiment already started; use a new output for a new experiment")
        references = [r for r in review["candidates"] if r["role"] == "references" and r.get("confirmed") is True]
        if not references:
            raise ValueError("Adam must confirm enrollment reference crops first")
        item = store.register("Adam's keys", "bunch of keys")
        for row in references:
            if row["clip"] not in review["split"]["references"] or row["review"] != "key":
                raise ValueError("reference split/review violation")
            cap = cv2.VideoCapture(str(Path(args.clips) / row["clip"]))
            cap.set(cv2.CAP_PROP_POS_MSEC, row["frame_ms"])
            ok, image = cap.read()
            cap.release()
            if not ok:
                raise ValueError("unreadable reference frame")
            store.add_example(item, image, row["box"], source=row["clip"], frame_ms=row["frame_ms"], confirmed=True)
        positives, negatives, records = [], [], []
        for row in review["candidates"]:
            if row["role"] != "development":
                continue
            if row["clip"] not in review["split"]["development"] or row["review"] not in ["key", "non_key"] or not row["reviewer"]:
                raise ValueError("all development candidates need explicit review and correct split")
            # Reconstruct the reviewed JPEG crop from the source recording when
            # only the shareable annotation manifest is present (no private images).
            crop_path = Path(args.review).parent / row["crop"]
            if crop_path.exists():
                crop = cv2.imread(str(crop_path))
            else:
                cap = cv2.VideoCapture(str(Path(args.clips) / row["clip"]))
                cap.set(cv2.CAP_PROP_POS_MSEC, row["frame_ms"])
                ok, image = cap.read()
                cap.release()
                if not ok:
                    raise ValueError("unreadable development frame")
                encoded = cv2.imencode(".jpg", crop_image(image, row["box"]))[1]
                crop = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            score = store.match(encoder.encode(crop), "bunch of keys", -1, .05)["score"]
            (positives if row["review"] == "key" else negatives).append(score)
            records.append({**row, "score": score})
        threshold = choose_threshold(positives, negatives)
        atomic_json(out / "frozen.json", dict(threshold=threshold, margin=.05, item_id=item,
                    encoder_version=encoder.version, review_sha256=sha(args.review),
                    reference_sha256=sha(out / "store" / item / "item.json"), development=records,
                    claim="exploratory single-item matching across recording sessions"))
        atomic_json(out / "bridge.json", dict(store=str((out / "store").resolve()),
                    checkpoint=str(Path(args.checkpoint).resolve()), threshold=threshold, margin=.05))
        print(f"Frozen threshold {threshold}; development accepted {sum(s >= threshold for s in positives)}/{len(positives)} keys; 0/{len(negatives)} negatives")
    else:
        from memory_engine.detector import YoloWorldDetector
        frozen = json.loads((out / "frozen.json").read_text())
        if frozen["encoder_version"] != encoder.version or frozen["review_sha256"] != sha(args.review):
            raise ValueError("frozen experiment inputs changed")
        if frozen["reference_sha256"] != sha(out / "store" / frozen["item_id"] / "item.json"):
            raise ValueError("frozen references changed")
        if (out / "heldout.json").exists():
            raise ValueError("heldout output exists; preserve its review")
        detector = YoloWorldDetector(["bunch of keys"], model=args.weights, imgsz=960, conf=.05,
                                    half=True, distractors=["laptop keyboard", "laptop"])
        rows, frames = [], []
        for name in review["split"]["held_out_positive"] + review["split"]["held_out_no_keys"]:
            cap = cv2.VideoCapture(str(Path(args.clips) / name))
            duration = cap.get(cv2.CAP_PROP_FRAME_COUNT)/cap.get(cv2.CAP_PROP_FPS)*1000
            for ms in range(0, int(duration), review["split"]["sample_interval_ms"]):
                cap.set(cv2.CAP_PROP_POS_MSEC, ms)
                ok, image = cap.read()
                if not ok:
                    continue
                actual_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                start = time.perf_counter()
                detections = detector.detect(image)
                detect_time = time.perf_counter()-start
                for det in detections:
                    crop = crop_image(image, det.bbox)
                    match = store.match(encoder.encode(crop), det.label, frozen["threshold"], frozen["margin"])
                    index = len(rows)
                    cv2.imwrite(str(out / f"heldout_{index:04d}.jpg"), crop)
                    rows.append(dict(id=index, clip=name, frame_ms=actual_ms, box=list(det.bbox),
                                     confidence=det.conf, review=None, reviewer=None, **match))
                frames.append(dict(clip=name, frame_ms=actual_ms, proposals=len(detections),
                                   detection_seconds=detect_time, total_seconds=time.perf_counter()-start))
            cap.release()
        atomic_json(out / "heldout.json", dict(frozen_sha256=sha(out / "frozen.json"), candidates=rows, frames=frames))
        print(f"Saved {len(rows)} held-out proposals for localization review")


if __name__ == "__main__":
    main()
