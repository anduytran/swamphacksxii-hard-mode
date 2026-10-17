"""Turn per-frame detections into object memories.

Three problems this solves:

1. One-frame flukes. A detector will sometimes "see" keys for a single frame. So a
   detection first becomes a *track* (the same object followed across frames by box
   overlap), and a track only becomes a *memory* once it has been seen in enough
   frames over enough time.

2. Duplicates vs. moves. When a new sighting is confirmed we ask: is this an object
   we already remember, in the same place?
   - Without a scene matcher (original behavior): "same place" = the box overlaps an
     existing memory's box in image coordinates. Only valid for a steady camera.
   - With a scene matcher (scene.py, the feature flag): compare the *surroundings* in
     the new frame with each remembered evidence image. If it's the same scene and the
     remembered box, mapped into the new frame, lands on the new detection, it's the
     same place (resighted). Otherwise a new memory is created and labeled honestly:
       place_status = "likely_new_place"  every comparison said different_scene
                      "moved_in_scene"    same scene, but the object is somewhere else in it
                      "unverified"        couldn't tell (low texture, unknown, untrusted projection)
                      "first_sighting"    nothing to compare against
                      "another_instance"  a second object with the same label, seen at the same time
     A failed match is never treated as proof of a new place; old memories are never
     marked "not there".

3. Stale evidence. The saved image must show where the object was *last* seen, not
   just the clearest frame from earlier. When a sighting ends, the evidence switches to
   the last frame where the detection was still reasonably confident.

The logic here is pure: no model, and OpenCV only through the optional scene matcher.
Events that need a picture saved carry it under the private key "_image"; store.py
removes that key before writing anything.
"""

import re
import uuid
from dataclasses import dataclass, field

from memory_engine.detector import Detection


@dataclass
class MemoryConfig:
    min_conf: float = 0.25        # ignore detections weaker than this
    match_iou: float = 0.3        # box overlap needed to say "same track as last frame"
    min_hits: int = 3             # frames a track needs before it becomes a memory...
    min_seen_ms: int = 500        # ...spread over at least this much time
    max_gap_ms: int = 1500        # a track not seen for this long has ended
    same_place_iou: float = 0.5   # (no scene matcher) overlap with a memory's box = same place
    better_conf_margin: float = 0.05  # mid-sighting: replace the image only if confidence beats it by this
    evidence_min_ratio: float = 0.6   # a frame can be end-of-sighting evidence if conf >= 60% of the track's best
    scene_max_candidates: int = 5     # (scene matcher) compare against at most this many recent memories
    scene_same_spot_iou: float = 0.3  # (scene matcher) projected old box vs new box overlap for "same spot"


@dataclass
class Evidence:
    det: Detection
    frame_id: int
    video_ms: int
    wall_ms: int
    observed_ms: int | None
    image: object = field(default=None, repr=False)


@dataclass
class Track:
    track_id: int
    label: str
    bbox: tuple
    first_ms: int
    last_ms: int
    hits: int = 1
    memory_id: str | None = None
    best_conf: float = 0.0
    last_good: Evidence | None = None  # most recent frame with conf >= evidence_min_ratio x best_conf


@dataclass
class Memory:
    memory_id: str
    label: str
    bbox: tuple
    confidence: float
    frame_id: int          # frame the saved image came from
    video_ms: int          # when that frame was taken (source-relative time)
    wall_ms: int           # when that frame was processed
    observed_ms: int | None  # real-world time that frame was seen (None if unknown)
    first_seen_video_ms: int
    last_seen_video_ms: int
    last_seen_wall_ms: int
    last_seen_observed_ms: int | None
    sightings: int = 1     # separate times we came back to this object in this place
    place_status: str | None = None   # only set when the scene matcher is on (see module docstring)
    scene_verdict: str | None = None  # best verdict from the comparison that decided place_status
    scene_inliers: int | None = None
    compared_with: str | None = None  # memory_id that comparison was against


def iou(a, b) -> float:
    """Intersection over union of two (x1, y1, x2, y2) boxes: 0 = no overlap, 1 = identical."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def same_spot(projected, bbox, min_iou) -> bool:
    """Projected old box and new box overlap, or their centers are within half a box size."""
    if iou(projected, bbox) >= min_iou:
        return True
    pcx, pcy = (projected[0] + projected[2]) / 2, (projected[1] + projected[3]) / 2
    bcx, bcy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    size = ((bbox[2] - bbox[0]) + (bbox[3] - bbox[1])) / 2
    return ((pcx - bcx) ** 2 + (pcy - bcy) ** 2) ** 0.5 <= 0.5 * size


def new_memory_id(label: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-")
    return f"mem_{slug}_{uuid.uuid4().hex[:8]}"


class MemoryMaker:
    def __init__(self, config: MemoryConfig | None = None, scene_matcher=None):
        """scene_matcher: a memory_engine.scene.SceneMatcher to decide "same place" from the
        surroundings, or None for the original image-position rule (the feature flag)."""
        self.config = config or MemoryConfig()
        self.scene = scene_matcher
        self.tracks: list[Track] = []
        self.memories: dict[str, Memory] = {}
        self._features = {}    # memory_id -> scene features of its evidence image (surroundings only)
        self._suppressed = {}  # label -> video_ms until which no new memory is created (after "found it")
        self._now_ms = 0
        self._next_track_id = 0

    def process(self, frame_id: int, video_ms: int, wall_ms: int, detections: list[Detection],
                image=None, observed_ms: int | None = None) -> list[dict]:
        """Feed one frame's detections.

        Returns events: created, image_updated, resighted, sighting_ended.
        """
        cfg = self.config
        self._now_ms = video_ms
        events = []
        detections = sorted((d for d in detections if d.conf >= cfg.min_conf), key=lambda d: -d.conf)

        # 1. Match each detection to an existing track (same label, overlapping box).
        matched = set()
        updated = []
        for det in detections:
            best, best_iou = None, cfg.match_iou
            for track in self.tracks:
                if track.label == det.label and track.track_id not in matched:
                    overlap = iou(track.bbox, det.bbox)
                    if overlap >= best_iou:
                        best, best_iou = track, overlap
            if best is None:
                best = Track(self._next_track_id, det.label, det.bbox, video_ms, video_ms)
                self._next_track_id += 1
                self.tracks.append(best)
            else:
                best.bbox, best.last_ms, best.hits = det.bbox, video_ms, best.hits + 1
            ev = Evidence(det, frame_id, video_ms, wall_ms, observed_ms, image)
            best.best_conf = max(best.best_conf, det.conf)
            if det.conf >= cfg.evidence_min_ratio * best.best_conf:
                best.last_good = ev
            matched.add(best.track_id)
            updated.append((best, ev))

        # 2. Confirm tracks into memories, or improve the memory they belong to.
        for track, ev in updated:
            if track.memory_id is None:
                if track.hits >= cfg.min_hits and video_ms - track.first_ms >= cfg.min_seen_ms \
                        and video_ms >= self._suppressed.get(track.label, -1):
                    events.append(self._confirm(track, ev))
            else:
                memory = self.memories[track.memory_id]
                self._touch(memory, ev)
                if ev.det.conf >= memory.confidence + cfg.better_conf_margin:
                    events.append(self._set_evidence(memory, ev, "image_updated"))

        # 3. End tracks we haven't seen for a while.
        for track in [t for t in self.tracks if video_ms - t.last_ms > cfg.max_gap_ms]:
            events += self._end(track)
        return events

    def finish(self) -> list[dict]:
        """Call when the source ends: closes every open sighting."""
        events = []
        for track in list(self.tracks):
            events += self._end(track)
        return events

    def forget(self, memory_id: str, cooldown_ms: int = 5000) -> bool:
        """Drop a memory (the user found the object).

        Tracks of that label are reset and no new memory of that label is created for
        cooldown_ms, so keys picked up and still in view don't instantly become a new memory.
        """
        memory = self.memories.pop(memory_id, None)
        self._features.pop(memory_id, None)
        if memory is None:
            return False
        self.tracks = [t for t in self.tracks if t.memory_id != memory_id and t.label != memory.label]
        self._suppressed[memory.label] = self._now_ms + cooldown_ms
        return True

    def _end(self, track: Track) -> list[dict]:
        self.tracks.remove(track)
        if not track.memory_id:
            return []
        memory = self.memories[track.memory_id]
        events = []
        last = track.last_good
        # Evidence must show the latest location: switch to the final reasonably confident frame
        # of this sighting, even if an older frame was clearer.
        if last and last.video_ms > memory.video_ms:
            events.append(self._set_evidence(memory, last, "image_updated"))
        events.append(self._event("sighting_ended", memory))
        return events

    def _confirm(self, track: Track, ev: Evidence) -> dict:
        if self.scene is not None and ev.image is not None:
            return self._confirm_by_scene(track, ev)
        for memory in self.memories.values():
            if memory.label == track.label and iou(memory.bbox, track.bbox) >= self.config.same_place_iou:
                return self._resight(track, memory, ev)
        return self._create(track, ev)

    def _confirm_by_scene(self, track: Track, ev: Evidence) -> dict:
        # A memory whose object is being tracked in this very frame is a *different* object
        # (two key rings in view), not a candidate for "same place" or "moved".
        in_view = {t.memory_id for t in self.tracks if t.memory_id and t.last_ms == ev.video_ms}
        same_label = [m for m in self.memories.values() if m.label == track.label and m.memory_id in self._features]
        candidates = sorted((m for m in same_label if m.memory_id not in in_view),
                            key=lambda m: m.last_seen_video_ms, reverse=True)[:self.config.scene_max_candidates]
        if not candidates:
            status = "another_instance" if same_label else "first_sighting"
            return self._create(track, ev, place_status=status)
        current = self.scene.features(ev.image, exclude_boxes=[ev.det.bbox])
        results = []
        for memory in candidates:
            r = self.scene.match(self._features[memory.memory_id], current)
            results.append((memory, r))
            if r.verdict == "same_scene" and r.projection_trusted:
                projected = self.scene.project_box(r, memory.bbox)
                if same_spot(projected, ev.det.bbox, self.config.scene_same_spot_iou):
                    memory.scene_verdict, memory.scene_inliers = r.verdict, r.inliers
                    return self._resight(track, memory, ev)
        # Not the same spot as anything we remember: new memory, labeled by what we know.
        same = [(m, r) for m, r in results if r.verdict == "same_scene" and r.projection_trusted]
        if same:
            status, (memory, r) = "moved_in_scene", same[0]
        elif all(r.verdict == "different_scene" for _, r in results):
            status, (memory, r) = "likely_new_place", max(results, key=lambda mr: mr[1].inliers)
        else:
            status, (memory, r) = "unverified", max(results, key=lambda mr: mr[1].inliers)
        return self._create(track, ev, place_status=status, scene_verdict=r.verdict,
                            scene_inliers=r.inliers, compared_with=memory.memory_id)

    def _resight(self, track: Track, memory: Memory, ev: Evidence) -> dict:
        """Same object, same place: reattach instead of duplicating."""
        track.memory_id = memory.memory_id
        memory.sightings += 1
        self._touch(memory, ev)
        if ev.det.conf >= memory.confidence + self.config.better_conf_margin:
            return self._set_evidence(memory, ev, "image_updated")
        return self._event("resighted", memory)

    def _create(self, track: Track, ev: Evidence, **place) -> dict:
        memory = Memory(new_memory_id(track.label), track.label, ev.det.bbox, ev.det.conf, ev.frame_id,
                        ev.video_ms, ev.wall_ms, ev.observed_ms, first_seen_video_ms=track.first_ms,
                        last_seen_video_ms=ev.video_ms, last_seen_wall_ms=ev.wall_ms,
                        last_seen_observed_ms=ev.observed_ms, **place)
        self.memories[memory.memory_id] = memory
        track.memory_id = memory.memory_id
        self._remember_scene(memory, ev)
        return self._event("created", memory, ev.image)

    def _remember_scene(self, memory: Memory, ev: Evidence):
        if self.scene is not None and ev.image is not None:
            self._features[memory.memory_id] = self.scene.features(ev.image, exclude_boxes=[memory.bbox])

    @staticmethod
    def _touch(memory: Memory, ev: Evidence):
        memory.last_seen_video_ms = ev.video_ms
        memory.last_seen_wall_ms = ev.wall_ms
        memory.last_seen_observed_ms = ev.observed_ms

    def _set_evidence(self, memory: Memory, ev: Evidence, kind: str) -> dict:
        memory.bbox, memory.confidence = ev.det.bbox, ev.det.conf
        memory.frame_id, memory.video_ms, memory.wall_ms, memory.observed_ms = (
            ev.frame_id, ev.video_ms, ev.wall_ms, ev.observed_ms)
        self._remember_scene(memory, ev)
        return self._event(kind, memory, ev.image)

    @staticmethod
    def _event(kind: str, memory: Memory, image=None) -> dict:
        event = {
            "event": kind,
            "memory_id": memory.memory_id,
            "label": memory.label,
            "bbox": list(memory.bbox),
            "confidence": memory.confidence,
            "frame_id": memory.frame_id,
            "video_ms": memory.video_ms,
            "wall_ms": memory.wall_ms,
            "observed_ms": memory.observed_ms,
            "first_seen_video_ms": memory.first_seen_video_ms,
            "last_seen_video_ms": memory.last_seen_video_ms,
            "last_seen_wall_ms": memory.last_seen_wall_ms,
            "last_seen_observed_ms": memory.last_seen_observed_ms,
            "sightings": memory.sightings,
            "place_status": memory.place_status,
            "scene_verdict": memory.scene_verdict,
            "scene_inliers": memory.scene_inliers,
            "compared_with": memory.compared_with,
        }
        if image is not None:
            event["_image"] = image
        return event
