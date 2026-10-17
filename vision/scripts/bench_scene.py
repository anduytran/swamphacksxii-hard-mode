"""How fast is the scene matcher, and where does it stop matching?

    python scripts/bench_scene.py

Part 1 times features() and match() on this machine.
Part 2 warps a real photo with increasingly random viewpoints and counts verdicts.
The warps are flat (planar), so part 2 is a best case, not a real-world accuracy number.
"""

import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from memory_engine.scene import SAME, SceneMatcher  # noqa: E402

FIX = Path(__file__).resolve().parents[1] / "tests" / "fixtures"


def median_ms(fn, reps=30):
    times = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t) * 1000)
    return statistics.median(times)


def main():
    m = SceneMatcher()
    bus = cv2.imread(str(FIX / "bus.jpg"))
    vga = cv2.resize(bus, (640, 480))  # roughly what an ESP32 camera sends
    a, b = m.features(bus), m.features(cv2.GaussianBlur(bus, (3, 3), 0))

    print("Part 1: timing (median of 30 runs)")
    print(f"  features(), 810x1080 photo : {median_ms(lambda: m.features(bus)):6.1f} ms")
    print(f"  features(), 640x480 frame  : {median_ms(lambda: m.features(vga)):6.1f} ms")
    print(f"  match(), one pair          : {median_ms(lambda: m.match(a, b)):6.1f} ms")

    print("\nPart 2: random viewpoint warps of one photo (40 per row, planar = best case)")
    print(f"  {'max corner shift':>16} {'same':>5} {'unknown':>8} {'different':>10} {'trusted':>8} {'worst trusted err px':>21}")
    rng = np.random.default_rng(1)
    h, w = bus.shape[:2]
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    box = (223, 406, 344, 860)
    corners = np.float32([[box[0], box[1]], [box[2], box[1]], [box[2], box[3]], [box[0], box[3]]]).reshape(-1, 1, 2)
    for severity in (0.05, 0.10, 0.20, 0.30, 0.40):
        counts = {"same_scene": 0, "unknown": 0, "different_scene": 0}
        trusted, worst = 0, 0.0
        for _ in range(40):
            H = cv2.getPerspectiveTransform(src, (src + rng.uniform(-severity, severity, (4, 2)) * [w, h]).astype(np.float32))
            r = m.match(a, m.features(cv2.warpPerspective(bus, H, (w, h))))
            counts[r.verdict] += 1
            if r.projection_trusted:
                trusted += 1
                gt = cv2.perspectiveTransform(corners, H).reshape(-1, 2)
                gt = (*gt.min(0), *gt.max(0))
                worst = max(worst, max(abs(p - q) for p, q in zip(m.project_box(r, box), gt)))
        print(f"  {severity:>15.0%} {counts[SAME]:>5} {counts['unknown']:>8} {counts['different_scene']:>10} "
              f"{trusted:>8} {worst:>21.1f}")


if __name__ == "__main__":
    main()
