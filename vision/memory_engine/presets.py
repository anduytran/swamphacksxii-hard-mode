"""Detector presets: one place that defines them, used by run.py, the LastSeen API and the eval pipeline.

Measured on our phone clips (README "Detector benchmark" and "Real-world evaluation, round 2"):

    efficient (default)  m, 480 px                     best on low-res close-ups (360x640 clips); fails on wide
                                                        1080p frames and reads laptop keyboards as "keys"
    light                s, 640 px                     least power, misses more
    xl                   x, 640 px                     best close-up recall on low-res clips, most power
    m480d                m, 480 px + distractors       efficient's resolution, without the keyboard false alarms (untested on clips)
    m640d                m, 640 px + distractors       candidate for 640x480 ESP32 frames (= native size)
    m960d                m, 960 px + distractors       wide 1080p frames: 0 false memories on the no-keys clip,
                                                        4/5 useful placement memories; worse on low-res frames
    x960d                x, 960 px + distractors       wide 1080p frames: 5/5 useful placement memories, 2x the GPU time

"Distractors" are extra classes that are detected but never returned ("laptop keyboard", "laptop").
With class-agnostic NMS, a keyboard box that out-scores an overlapping "bunch of keys" box
suppresses it. Possible side effect: keys lying on or against a laptop could be suppressed too.
All presets run 2 inferences/s with min_conf 0.05 and FP16.
"""

DISTRACTORS = ("laptop keyboard", "laptop")

_BASE = {"half": True, "inference_fps": 2.0, "min_conf": 0.05, "distractors": ()}

PRESETS = {
    "efficient": {**_BASE, "model": "yolov8m-worldv2.pt", "imgsz": 480},
    "light": {**_BASE, "model": "yolov8s-worldv2.pt", "imgsz": 640},
    "xl": {**_BASE, "model": "yolov8x-worldv2.pt", "imgsz": 640},
    "m480d": {**_BASE, "model": "yolov8m-worldv2.pt", "imgsz": 480, "distractors": DISTRACTORS},
    "m640d": {**_BASE, "model": "yolov8m-worldv2.pt", "imgsz": 640, "distractors": DISTRACTORS},
    "m960d": {**_BASE, "model": "yolov8m-worldv2.pt", "imgsz": 960, "distractors": DISTRACTORS},
    "x960d": {**_BASE, "model": "yolov8x-worldv2.pt", "imgsz": 960, "distractors": DISTRACTORS},
}
DEFAULT_PRESET = "efficient"


def make_detector(preset: str | dict, prompts, conf: float | None = None, **overrides):
    """Build a YoloWorldDetector from a preset name (or dict). conf defaults to min(min_conf, 0.10)."""
    from memory_engine.detector import YoloWorldDetector

    p = {**(PRESETS[preset] if isinstance(preset, str) else preset), **overrides}
    return YoloWorldDetector(list(prompts), model=p["model"], imgsz=p["imgsz"], half=p["half"],
                             conf=min(p["min_conf"], 0.10) if conf is None else conf,
                             distractors=p["distractors"])
