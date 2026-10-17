"""The ML interface Dylan's backend calls. Plain Python, in-process, no web server.

Dylan owns the HTTP/WebSocket endpoints, voice, UI and camera plumbing; each endpoint
can be a thin wrapper around one of these methods.

    from memory_engine.api import LastSeen, resolve_label

    ls = LastSeen(out_dir="data/memories_out")            # efficient preset, scene matching on
    events = ls.process_frame(frame_bgr, video_ms)          # call for each frame from the camera loop
    mem = ls.latest_memory(resolve_label("my keys"))       # "where are my keys?"
    g = ls.guide(frame_bgr, mem["memory_id"])              # while the user looks around: where to look
    hit = ls.locate(frame_bgr, mem["label"], search=True)  # on-demand thorough search (XL + tiles)
    ls.found(mem["memory_id"])                              # "found it": deletes image, crop, record, history

Every method returns plain dicts/lists (JSON-serializable). Boxes are [x1, y1, x2, y2]
in pixels of the image they refer to. Directions are relative to the camera image:
offset_x = -1 (left edge) .. 0 (center) .. +1 (right edge); beyond +-1 = outside the frame.
"""

from pathlib import Path

import cv2

from memory_engine.memory import MemoryConfig, MemoryMaker, same_spot
from memory_engine.scene import SceneMatcher, visible_fraction
from memory_engine.store import MemoryStore

ALIASES = {"keys": "bunch of keys", "key": "bunch of keys", "my keys": "bunch of keys",
           "keychain": "bunch of keys", "car keys": "bunch of keys"}

CAVEATS = {
    "unverified": "Couldn't confirm whether this is a new spot or a different view of an earlier one.",
    "moved_in_scene": "Seen in a different spot than before, in the same area.",
    "another_instance": "More than one of these was in view at the same time.",
}


def resolve_label(text: str) -> str:
    """Map what the user says to the detector prompt used for memories."""
    t = text.strip().lower()
    return ALIASES.get(t, t)


def direction(box, shape, deadband=0.15) -> dict:
    """Camera-relative direction toward a box: left/right and up/down, with a center deadband."""
    h, w = shape[:2]
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    ox, oy = (cx - w / 2) / (w / 2), (cy - h / 2) / (h / 2)
    horizontal = "left" if ox < -deadband else "right" if ox > deadband else "center"
    vertical = "up" if oy < -deadband else "down" if oy > deadband else "middle"
    return {"offset_x": round(ox, 2), "offset_y": round(oy, 2), "horizontal": horizontal, "vertical": vertical,
            "in_frame": -1 <= ox <= 1 and -1 <= oy <= 1}


class LastSeen:
    def __init__(self, out_dir="data/memories_out", preset="efficient", scene_matching=True,
                 prompts=("bunch of keys",), source="live", session_id="", source_kind="live",
                 detector=None, search_detector=None, memory_config: MemoryConfig | None = None,
                 search_preset="xl"):
        """preset / search_preset: names from memory_engine/presets.py (search loads on first use).
        detector / search_detector can be injected instead (tests, or sharing a loaded model)."""
        from memory_engine.presets import PRESETS
        from memory_engine.run import memory_config as scaled_config

        self.preset = PRESETS[preset]
        self.search_preset = search_preset
        self.prompts = list(prompts)
        self._detector = detector
        self._search_detector = search_detector
        self.matcher = SceneMatcher()
        if memory_config is None:
            class _Args:  # reuse run.py's rate-scaled thresholds
                min_conf, min_hits, min_seen_ms, max_gap_ms = self.preset["min_conf"], None, None, None
            memory_config = scaled_config(_Args, self.preset["inference_fps"])
        self.maker = MemoryMaker(memory_config, scene_matcher=self.matcher if scene_matching else None)
        self.store = MemoryStore(out_dir, source, session_id=session_id, source_kind=source_kind)
        self._feature_cache = {}
        self._frame_id = 0

    # --- detectors (lazy) ---------------------------------------------------
    @property
    def detector(self):
        if self._detector is None:
            from memory_engine.presets import make_detector

            self._detector = make_detector(self.preset, self.prompts)
        return self._detector

    @property
    def search_detector(self):
        if self._search_detector is None:
            from memory_engine.presets import make_detector

            self._search_detector = make_detector(self.search_preset, self.prompts, conf=0.03)
        return self._search_detector

    # --- 1. memory creation --------------------------------------------------
    def process_frame(self, image, video_ms: int, wall_ms: int | None = None, observed_ms: int | None = None,
                      detections=None) -> list[dict]:
        """Detect, update memories, save evidence. Call at the inference rate (default 2/s)."""
        import time

        wall_ms = wall_ms if wall_ms is not None else int(time.time() * 1000)
        dets = self.detector.detect(image) if detections is None else detections
        events = [self.store.handle(e) for e in
                  self.maker.process(self._frame_id, video_ms, wall_ms, dets, image=image, observed_ms=observed_ms)]
        self._frame_id += 1
        return events

    def finish(self) -> list[dict]:
        return [self.store.handle(e) for e in self.maker.finish()]

    # --- 2. retrieval ----------------------------------------------------------
    def memories(self, label: str | None = None) -> list[dict]:
        """All stored memories (optionally for one label), most recent sighting first."""
        records = [r for r in self.store.records.values() if label is None or r["label"] == label]
        return sorted(records, key=_recency, reverse=True)

    def latest_memory(self, label: str) -> dict | None:
        """The most recent sighting of `label`, with caveats and older alternatives.

        Recency wins over confidence: if the keys were seen on the desk and later on the
        shelf, the shelf is the answer even if the desk photo was clearer.
        """
        ranked = self.memories(label)
        if not ranked:
            return None
        best = dict(ranked[0])
        caveats = [CAVEATS[best["place_status"]]] if best.get("place_status") in CAVEATS else []
        if best.get("confidence", 1) < 0.1:
            caveats.append("Low detection confidence for this sighting.")
        best.update({
            "image_file": str(self.store.out / best["image_path"]) if best.get("image_path") else None,
            "crop_file": str(self.store.out / best["crop_path"]) if best.get("crop_path") else None,
            "time_basis": "observed" if best.get("last_seen_observed_ms") is not None else "processing",
            "caveats": caveats,
            "alternatives": [r["memory_id"] for r in ranked[1:3]],
        })
        return best

    # --- 3. navigation ---------------------------------------------------------
    def locate(self, image, label: str, search: bool = False) -> dict:
        """Is the object visible in this live frame, and which way is it?

        search=True runs the XL model on the frame plus 2x2 tiles (finds smaller/farther
        objects, roughly 5x the cost of a normal XL call): only for user-requested searches.
        """
        if search:
            from memory_engine.detector import detect_tiled

            dets, model = detect_tiled(self.search_detector, image), "xl+tiles"
        else:
            dets, model = self.detector.detect(image), "default"
        dets = [d for d in dets if d.label == label]
        if not dets:
            return {"visible": False, "label": label, "model": model}
        best = max(dets, key=lambda d: d.conf)
        return {"visible": True, "label": label, "model": model, "bbox": list(best.bbox),
                "confidence": best.conf, **direction(best.bbox, image.shape)}

    def guide(self, image, memory_id: str, search: bool = False) -> dict:
        """Help the user get from "the area in the photo" to "the object".

        Compares the live frame's surroundings with the memory's evidence image. If they
        match, the remembered object box is mapped into the live frame, so the UI can draw
        "it was here" even when the detector can't see the object. A live detection is
        reported alongside. Never claims the object is gone: "not detected" can be a miss.
        """
        record = self.store.records.get(memory_id)
        if record is None:
            return {"status": "no_such_memory", "memory_id": memory_id}
        live = self.locate(image, record["label"], search=search)
        result = self.matcher.match(self._memory_features(record), self.matcher.features(image))
        out = {"memory_id": memory_id, "scene_verdict": result.verdict, "scene_reason": result.reason,
               "scene_inliers": result.inliers, "projection_trusted": result.projection_trusted,
               "detection": live, "remembered_box_in_view": None, "visible_fraction": None}
        projected = self.matcher.project_box(result, record["bbox"])
        if projected:
            out["remembered_box_in_view"] = list(projected)
            out["visible_fraction"] = round(visible_fraction(projected, image.shape), 2)
        if projected and live["visible"] and same_spot(projected, live["bbox"], 0.3):
            status, target = "object_at_remembered_spot", live["bbox"]
        elif live["visible"]:
            status, target = ("object_seen_elsewhere" if projected else "object_detected"), live["bbox"]
        elif projected and out["visible_fraction"] >= 0.5:
            status, target = "remembered_spot_in_view", projected
        elif projected:
            status, target = "remembered_spot_out_of_frame", projected
        else:
            status, target = "remembered_place_not_recognized", None
        out["status"] = status
        out["target_box"] = list(target) if target else None
        out["direction"] = direction(target, image.shape) if target else None
        return out

    # --- 4. recovery -----------------------------------------------------------
    def found(self, memory_id: str | None = None, label: str | None = None, cooldown_ms: int = 5000) -> list[dict]:
        """The user confirmed recovery: delete the memory (or every memory of `label`).

        Removes both images, the record, its event history, and cached scene features, and
        pauses new memories of that label for cooldown_ms (the keys are now in hand).
        """
        ids = [memory_id] if memory_id else [r["memory_id"] for r in self.memories(label)]
        removed = []
        for mid in ids:
            self.maker.forget(mid, cooldown_ms=cooldown_ms)
            self._feature_cache = {k: v for k, v in self._feature_cache.items() if k[0] != mid}
            info = self.store.delete(mid)
            if info:
                removed.append(info)
        return removed

    def close(self):
        self.store.close()

    def _memory_features(self, record):
        key = (record["memory_id"], record.get("video_ms"), record.get("image_path"))
        if key not in self._feature_cache:
            image = cv2.imread(str(Path(self.store.out) / record["image_path"]))
            self._feature_cache[key] = self.matcher.features(image, exclude_boxes=[record["bbox"]])
        return self._feature_cache[key]


def _recency(record):
    observed = record.get("last_seen_observed_ms")
    return observed if observed is not None else record.get("last_seen_wall_ms") or 0
