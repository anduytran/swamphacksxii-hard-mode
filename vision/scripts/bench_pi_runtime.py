"""Compare runtime settings on the device, using one unchanged real camera frame.

Run configurations in separate processes for independent peak RAM measurements.
This measures speed and numerical repeatability, not detection recall.
"""
import argparse
import hashlib
import json
import platform
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from lastscene_pipeline.common import resolve_model_dir
from lastscene_pipeline.bench import peak_rss_mb
from lastscene_pipeline.runtime import OnnxYoloDetector, letterbox, sha256


def temperature():
    try:
        return int(Path('/sys/class/thermal/thermal_zone0/temp').read_text()) / 1000
    except (OSError, ValueError):
        return None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--image', type=Path, required=True)
    p.add_argument('--threads', type=int, default=3)
    p.add_argument('--opencv-threads', type=int)
    p.add_argument('--runs', type=int, default=10)
    p.add_argument('--interval', type=float, default=1.0)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.runs < 2 or a.threads < 1 or a.interval < 0:
        p.error('runs must be >= 2, threads >= 1, and interval >= 0')
    if a.opencv_threads is not None:
        if a.opencv_threads < 1:
            p.error('opencv-threads must be >= 1')
        cv2.setNumThreads(a.opencv_threads)
    model_dir = resolve_model_dir(a.model)
    if model_dir is None:
        p.error('model directory cannot be resolved')
    frame = cv2.imread(str(a.image))
    if frame is None:
        p.error('image cannot be decoded')
    model = model_dir / 'model-int8.onnx'
    temp_before = temperature()
    t = time.perf_counter()
    det = OnnxYoloDetector(model, confidence=0.35, threads=a.threads, low_memory=True)
    load_s = time.perf_counter() - t
    det.detect(frame)
    times, signatures = [], []
    cpu_start, start = time.process_time(), time.perf_counter()
    for _ in range(a.runs):
        t = time.perf_counter()
        found = det.detect(frame)
        elapsed = time.perf_counter() - t
        times.append(elapsed)
        signatures.append([(d.label, float(d.conf), list(d.bbox)) for d in found])
        time.sleep(max(0, a.interval - (time.perf_counter() - t)))
    wall_s, cpu_s = time.perf_counter() - start, time.process_time() - cpu_start
    # Compare the raw output too: equal empty detection lists alone prove little.
    tensor, _ = letterbox(frame, det.size)
    output = det.session.run(None, {det.input.name: tensor})[0]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(a.out.with_suffix('.npy'), output)
    report = dict(
        model_version=model_dir.name, model_sha256=sha256(model),
        runtime_source_sha256=sha256(Path(sys.modules[OnnxYoloDetector.__module__].__file__)),
        image_sha256=sha256(a.image), frame_shape=list(frame.shape),
        model_input=det.size, ort_threads=a.threads, opencv_threads=cv2.getNumThreads(),
        runs=a.runs, interval_s=a.interval, load_s=load_s,
        median_s=statistics.median(times), p95_s=float(np.percentile(times, 95)),
        times_s=times, wall_s=wall_s, effective_fps=a.runs/wall_s,
        process_cpu_s=cpu_s, mean_cpu_percent=100*cpu_s/wall_s,
        peak_rss_mb=peak_rss_mb(),
        temperature_before_c=temp_before, temperature_after_c=temperature(),
        detections=signatures[-1], detections_stable=all(s == signatures[0] for s in signatures),
        raw_output_sha256=hashlib.sha256(output.tobytes()).hexdigest(),
        python=platform.python_version(), machine=platform.machine(),
        onnxruntime=sys.modules['onnxruntime'].__version__, opencv=cv2.__version__,
        limitation='One fixed camera frame; no recall or false-memory evaluation.',
    )
    a.out.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
