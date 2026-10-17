"""Convert LastSeen memory events into the contract's observation (JSON + full-frame image).

Contract fields (POST /api/v1/observations, multipart):
    image        full uncropped frame, JPEG or PNG, <= 2 MB
    observation  JSON: observation_id, captured_at, source="auto", pipeline{name,version,detector,matcher},
                 image_size{width,height}, scene (optional), objects[{label, confidence, bbox[x,y,w,h], track_id, event}]

Conservative choices (see README_HANDOFF "Decisions for Dylan"):
- event is always "seen": we confirm an object is there, but never prove placed/moved/picked_up.
- scene is omitted: ORB gives "same/different as that memory", not a stable room id or human label.
- captured_at comes from observed_ms (real-world time). If it's unknown, there is no observation:
  NoCaptureTime is raised instead of guessing.
- confidence is the raw detector confidence, even below the contract's 0.35 (flagged, not dropped).
"""

import datetime
import uuid

import cv2
import numpy as np

from lastscene_bridge import BRIDGE_VERSION, PIPELINE_NAME

MAX_IMAGE_BYTES = 2 * 1024 * 1024
CONTRACT_MIN_CONFIDENCE = 0.35
# detector prompt -> item name the contract expects (lowercase singular)
LABEL_TO_ITEM = {"bunch of keys": "keys", "keys": "keys", "keychain": "keys", "key ring": "keys"}
_NAMESPACE = uuid.UUID("5d7c3b1e-2f4a-4e8b-9c6d-1a2b3c4d5e6f")  # fixed: same frame -> same observation_id


class NoCaptureTime(ValueError):
    """The real-world capture time is unknown (e.g. a recording without metadata)."""


def item_label(prompt_label: str) -> str:
    return LABEL_TO_ITEM.get(prompt_label.strip().lower(), prompt_label.strip().lower())


def captured_at_iso(observed_ms: int) -> str:
    dt = datetime.datetime.fromtimestamp(observed_ms / 1000, tz=datetime.timezone.utc)
    return dt.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def observation_id(source: str, observed_ms: int) -> str:
    """Stable id for one frame: retries and re-runs on the same clip reuse it (server treats repeats as no-ops)."""
    return str(uuid.uuid5(_NAMESPACE, f"{source}|{observed_ms}"))


def normalize_bbox(bbox_px, width: int, height: int) -> list[float]:
    """[x1, y1, x2, y2] pixels -> [x, y, w, h] normalized 0-1, top-left origin, clipped to the image."""
    x1, y1, x2, y2 = (float(v) for v in bbox_px)
    x1, x2 = max(0.0, min(x1, width)), max(0.0, min(x2, width))
    y1, y2 = max(0.0, min(y1, height)), max(0.0, min(y2, height))
    return [round(x1 / width, 4), round(y1 / height, 4), round((x2 - x1) / width, 4), round((y2 - y1) / height, 4)]


def prepare_image(image_bytes: bytes, max_bytes: int = MAX_IMAGE_BYTES) -> tuple[bytes, int, int, float]:
    """Return (jpeg_bytes, width, height, scale) within max_bytes: re-encode, then downscale only if needed.

    width/height are of the image actually uploaded; scale = uploaded width / original width,
    for converting pixel boxes measured on the original evidence image.
    """
    img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("evidence image could not be decoded")
    h, w = img.shape[:2]
    if len(image_bytes) <= max_bytes and image_bytes[:2] == b"\xff\xd8":
        return image_bytes, w, h, 1.0
    for quality in (90, 80, 70):
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if ok and len(buf) <= max_bytes:
            return buf.tobytes(), w, h, 1.0
    small = img
    while True:
        small = cv2.resize(small, None, fx=0.75, fy=0.75, interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok and len(buf) <= max_bytes:
            return buf.tobytes(), small.shape[1], small.shape[0], small.shape[1] / w


def pipeline_info(preset_name: str, preset: dict, scene_matching: bool = True) -> dict:
    model = preset["model"].replace(".pt", "")
    extra = "+distractors" if preset.get("distractors") else ""
    return {"name": PIPELINE_NAME, "version": BRIDGE_VERSION,
            "detector": f"yolo-world {model} {preset['imgsz']}px{extra} (preset {preset_name})",
            "matcher": "orb" if scene_matching else "none"}


def build_observation(records: list[dict], image_bytes: bytes, pipeline: dict) -> tuple[dict, bytes]:
    """records: memory events/records that share one evidence frame (same source + observed_ms).

    Returns (observation_dict, jpeg_bytes_to_upload).
    """
    if not records:
        raise ValueError("no records")
    first = records[0]
    observed = first.get("observed_ms")
    if observed is None:
        raise NoCaptureTime(f"{first['memory_id']}: observed_ms is null (recording time unknown); not uploading")
    jpeg, w, h, scale = prepare_image(image_bytes)
    objects = [{
        "label": item_label(r["label"]),
        "confidence": round(float(r["confidence"]), 4),
        "bbox": normalize_bbox([v * scale for v in r["bbox"]], w, h),  # box on the uploaded image, / its size
        "track_id": r["memory_id"],
        "event": "seen",
    } for r in records]
    obs = {
        "observation_id": observation_id(first.get("source", ""), observed),
        "captured_at": captured_at_iso(observed),
        "source": "auto",
        "pipeline": pipeline,
        "image_size": {"width": w, "height": h},
        "objects": objects,
    }
    return obs, jpeg


def below_contract_threshold(obs: dict) -> list[dict]:
    """Objects under the contract's proposed 0.35 minimum (sent anyway; reported, not dropped)."""
    return [o for o in obs["objects"] if o["confidence"] < CONTRACT_MIN_CONFIDENCE]
