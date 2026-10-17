"""Prepare unconfirmed reference/development crop candidates; never enroll automatically.

Run as python -m scripts.prepare_enrollment from vision/. Images stay under data/.
Review review.json: set each candidate review to key or non_key and reviewer to
the actual reviewer. Enrollment references additionally require Adam's confirmation.
"""
import argparse
import json
from pathlib import Path
import time

import cv2
from memory_engine.detector import YoloWorldDetector
from memory_engine.enrollment import atomic_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--clips", required=True)
    p.add_argument("--weights", required=True)
    p.add_argument("--out", default="data/enrollment_review")
    p.add_argument("--split", default="eval_manifests/enrollment_sessions.json")
    args = p.parse_args()
    split = json.loads(Path(args.split).read_text())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "review.json").exists():
        raise ValueError("review already exists; use a new output directory")
    detector = YoloWorldDetector(["bunch of keys"], model=args.weights, imgsz=960, conf=.05,
                                 half=True, distractors=["laptop keyboard", "laptop"])
    rows, frames = [], []
    for role in ["references", "development"]:
        for name in split[role]:
            cap = cv2.VideoCapture(str(Path(args.clips) / name))
            duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS) * 1000
            for ms in range(0, int(duration), split["sample_interval_ms"]):
                cap.set(cv2.CAP_PROP_POS_MSEC, ms)
                ok, image = cap.read()
                if not ok:
                    continue
                actual_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                start = time.perf_counter()
                detections = detector.detect(image)
                frames.append(dict(clip=name, frame_ms=actual_ms, count=len(detections),
                                   detection_seconds=time.perf_counter()-start))
                for det in detections:
                    row_id = len(rows)
                    x1, y1, x2, y2 = det.bbox
                    crop = image[y1:y2, x1:x2]
                    if not crop.size:
                        continue
                    cv2.imwrite(str(out / f"{row_id:04d}.jpg"), crop)
                    rows.append(dict(id=row_id, role=role, clip=name, frame_ms=actual_ms,
                                     box=list(det.bbox), confidence=det.conf, review=None,
                                     reviewer=None, confirmed=False, crop=f"{row_id:04d}.jpg"))
            cap.release()
    atomic_json(out / "review.json", dict(split=split, candidates=rows, frames=frames))
    from PIL import Image, ImageDraw, ImageOps
    for offset in range(0, len(rows), 30):
        batch = rows[offset:offset+30]
        sheet = Image.new("RGB", (1000, ((len(batch)+4)//5)*180), "white")
        draw = ImageDraw.Draw(sheet)
        for j, row in enumerate(batch):
            x, y = j % 5 * 200, j // 5 * 180
            tile = ImageOps.contain(Image.open(out / row["crop"]), (190, 135))
            sheet.paste(tile, (x, y))
            draw.text((x, y+136), f'{row["id"]}: {row["clip"][:22]}', fill="black")
            draw.text((x, y+153), f'{row["frame_ms"]:.0f} ms / {row["confidence"]}', fill="black")
        sheet.save(out / f"sheet_{offset//30}.jpg")
    print(f"Prepared {len(rows)} UNCONFIRMED crops in {out}")


if __name__ == "__main__":
    main()
