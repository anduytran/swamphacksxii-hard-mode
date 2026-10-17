"""Compare fixed-vocabulary FP32/INT8 with identical preprocessing and memory logic.

Frame-level recall requires reviewed localization labels. Raw proposal counts are
not recall. Store crops for that review and report memory counts separately.
"""
import argparse
import json
from pathlib import Path
import time

import cv2
import numpy as np
from memory_engine.api import LastSeen
from memory_engine.enrollment import atomic_json
from memory_engine.onnx_detector import OnnxDetector
from lastscene_bridge.capture_time import recording_time_ms
from scripts.export_int8 import sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--clips", required=True)
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--out", default="data/onnx_comparison")
    p.add_argument("--split", default="eval_manifests/enrollment_sessions.json")
    args = p.parse_args()
    split = json.loads(Path(args.split).read_text())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    reports = []
    for model in args.models:
        detector = OnnxDetector(model)
        detector.detect(np.zeros((480, 640, 3), np.uint8))
        report = dict(model=str(model), sha256=sha(model), bytes=Path(model).stat().st_size, clips={})
        for name in split["held_out_positive"]+split["held_out_no_keys"]:
            path = Path(args.clips) / name
            cap = cv2.VideoCapture(str(path))
            fps = cap.get(cv2.CAP_PROP_FPS)
            if not fps:
                raise ValueError(f"cannot read {name}")
            duration = cap.get(cv2.CAP_PROP_FRAME_COUNT)/fps*1000
            directory = out / Path(model).stem / Path(name).stem
            if directory.exists():
                raise ValueError("comparison output already exists; use a fresh output directory")
            directory.mkdir(parents=True)
            ls = LastSeen(out_dir=directory / "memories", preset="m960d", detector=detector,
                          search_detector=detector, source=str(path), source_kind="file")
            capture = recording_time_ms(path)
            frames, rows, times = [], [], []
            for ms in range(0, int(duration), split["sample_interval_ms"]):
                cap.set(cv2.CAP_PROP_POS_MSEC, ms)
                ok, image = cap.read()
                if not ok:
                    continue
                actual_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                start = time.perf_counter()
                detections = detector.detect(image)
                times.append(time.perf_counter()-start)
                ls.process_frame(image, int(actual_ms), observed_ms=int(capture+actual_ms) if capture else None,
                                 detections=detections)
                for det in detections:
                    index = len(rows)
                    x1,y1,x2,y2 = det.bbox
                    cv2.imwrite(str(directory / f"{index:04d}.jpg"), image[y1:y2,x1:x2])
                    rows.append(dict(id=index, frame_ms=actual_ms, box=list(det.bbox), confidence=det.conf,
                                     review=None, reviewer=None))
                frames.append(dict(frame_ms=actual_ms, proposals=len(detections)))
            ls.finish()
            memories = ls.memories("bunch of keys")
            ls.close()
            cap.release()
            result = dict(frames=frames, candidates=rows, memory_count=len(memories),
                          median_seconds=float(np.median(times)), p95_seconds=float(np.percentile(times,95)))
            atomic_json(directory / "review.json", result)
            report["clips"][name] = {k:v for k,v in result.items() if k not in ["frames", "candidates"]} | dict(frames=len(frames), proposals=len(rows))
        reports.append(report)
        atomic_json(out / "summary.json", reports)
        print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
