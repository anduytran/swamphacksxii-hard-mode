"""Thin wrapper around YOLO-World so the rest of the engine never touches Ultralytics."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Detection:
    label: str
    conf: float
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2 in pixels


def drop_oversized(detections, image_shape, max_area_frac):
    """Remove boxes covering more than max_area_frac of the image.

    At low confidence thresholds, open-vocabulary detectors sometimes return a box around
    the whole frame (seen on real clips: "bunch of keys" at conf 0.08 covering everything).
    Small handheld objects are never that big, so these are treated as noise.
    """
    h, w = image_shape[:2]
    limit = max_area_frac * h * w
    return [d for d in detections if (d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1]) <= limit]


class YoloWorldDetector:
    def __init__(self, prompts, model="yolov8s-worldv2.pt", conf=0.10, imgsz=640, max_area_frac=0.5,
                 half=False, distractors=()):
        """distractors: extra classes that are detected but never returned (e.g. "laptop keyboard").
        With them, NMS runs class-agnostic, so a keyboard box that out-scores an overlapping
        "bunch of keys" box suppresses it. On our clips this removed keyboard-as-keys false alarms."""
        from ultralytics import YOLO  # lazy import: tests of the pure logic don't need torch

        self.prompts = list(prompts)
        self.distractors = [d for d in distractors if d not in self.prompts]
        self.conf = conf
        self.imgsz = imgsz
        self.max_area_frac = max_area_frac
        self.half = half  # FP16 inference (CUDA only): less GPU work per frame
        self.model = YOLO(model)
        self.model.set_classes(self.prompts + self.distractors)

    def detect(self, image) -> list[Detection]:
        # Ultralytics 8.4 replaced `half=True` with `quantize=16` (FP16 compute on CUDA)
        precision = {"quantize": 16} if self.half else {}
        result = self.model.predict(image, conf=self.conf, imgsz=self.imgsz, verbose=False,
                                    agnostic_nms=bool(self.distractors), **precision)[0]
        detections = [
            Detection(
                label=self.prompts[int(box.cls)],
                conf=round(float(box.conf), 3),
                bbox=tuple(int(v) for v in box.xyxy[0].tolist()),
            )
            for box in result.boxes
            if int(box.cls) < len(self.prompts)  # distractor boxes are dropped here
        ]
        return drop_oversized(detections, image.shape, self.max_area_frac)


def detect_tiled(detector, image, grid=2, overlap=0.25) -> list[Detection]:
    """Run the detector on the whole frame plus a grid of overlapping tiles.

    Each tile is upscaled to the model's input size, so small/far objects get more pixels.
    On our clips this found the far-away keys in 7/30 frames (XL) where whole-frame
    detection found 0/30, at (grid^2 + 1) times the cost. Meant for on-demand searches only.
    """
    h, w = image.shape[:2]
    th, tw = int(h / grid * (1 + overlap)), int(w / grid * (1 + overlap))
    found = list(detector.detect(image))
    for r in range(grid):
        for c in range(grid):
            y0, x0 = min(int(r * h / grid), h - th), min(int(c * w / grid), w - tw)
            for d in detector.detect(image[y0:y0 + th, x0:x0 + tw]):
                x1, y1, x2, y2 = d.bbox
                found.append(Detection(d.label, d.conf, (x1 + x0, y1 + y0, x2 + x0, y2 + y0)))
    return found


class CascadeDetector:
    """Cheap model by default; the expensive model only when it's likely to matter.

    Per frame:
    - the light model finds a confident box (>= accept_conf): use it, done;
    - the light model finds a weak candidate (>= escalate_conf), or a periodic check is
      due (every `periodic_every` frames, 0 = never): run the heavy model on this frame
      and use its result instead;
    - otherwise: use the light model's (empty or weak) result.
    `force_heavy=True` on a call (e.g. a user-requested search) always runs the heavy model.
    """

    def __init__(self, light: YoloWorldDetector, heavy: YoloWorldDetector,
                 accept_conf=0.25, escalate_conf=0.03, periodic_every=0):
        self.light, self.heavy = light, heavy
        self.accept_conf, self.escalate_conf, self.periodic_every = accept_conf, escalate_conf, periodic_every
        self.frames = 0
        self.heavy_calls = 0

    def detect(self, image, force_heavy=False) -> list[Detection]:
        self.frames += 1
        if not force_heavy:
            light = self.light.detect(image)
            best = max((d.conf for d in light), default=0.0)
            periodic = self.periodic_every and self.frames % self.periodic_every == 0
            if best >= self.accept_conf or (best < self.escalate_conf and not periodic):
                return light
        self.heavy_calls += 1
        return self.heavy.detect(image)

    @property
    def heavy_fraction(self) -> float:
        return self.heavy_calls / self.frames if self.frames else 0.0
