"""Glue: LastSeen memory events -> selected observations -> uploader queue (or a dry-run outbox).

    bridge = LastSceneBridge(ls, uploader=uploader_from_env(...), preset_name="m960d")
    for frame in frames:
        bridge.handle(ls.process_frame(frame.image, frame.video_ms, frame.wall_ms, frame.observed_ms))
    bridge.handle(ls.finish())
    bridge.found("keys")      # local deletion + queued POST /api/v1/items/keys/found
"""

import datetime
import json
import time
from pathlib import Path

import cv2

from lastscene_bridge.observation import (CONTRACT_MIN_CONFIDENCE, NoCaptureTime, below_contract_threshold,
                                          build_observation, item_label, pipeline_info)
from lastscene_bridge.selector import EventSelector
from memory_engine.presets import PRESETS


class LastSceneBridge:
    def __init__(self, ls, uploader=None, preset_name="efficient", outbox_dir=None,
                 enforce_contract_min_confidence=False, selector: EventSelector | None = None):
        """uploader=None and outbox_dir set -> dry run: observations are written to disk, nothing is sent."""
        self.ls = ls
        self.uploader = uploader
        self.outbox = Path(outbox_dir) if outbox_dir else None
        if self.outbox:
            self.outbox.mkdir(parents=True, exist_ok=True)
        self.pipeline = pipeline_info(preset_name, PRESETS[preset_name], ls.maker.scene is not None)
        self.enforce = enforce_contract_min_confidence
        self.selector = selector or EventSelector()
        self.stats = {"events": 0, "observations": 0, "skipped_no_capture_time": 0,
                      "below_contract_min_confidence": 0, "dropped_by_contract_min_confidence": 0}
        self.sent_ids = {}  # memory_id -> last observation_id (for found)

    def handle(self, events: list[dict]) -> list[str]:
        self.stats["events"] += len(events)
        for e in events:
            if e.get("image_path") and e["event"] in ("created", "image_updated"):
                path = Path(self.ls.store.out) / e["image_path"]
                img = cv2.imread(str(path))
                if img is not None:
                    e["_image_size"] = (img.shape[1], img.shape[0])
        ids = []
        for group in self.selector.select(events):
            if self.enforce:
                kept = [r for r in group if r["confidence"] >= CONTRACT_MIN_CONFIDENCE]
                self.stats["dropped_by_contract_min_confidence"] += len(group) - len(kept)
                group = kept
                if not group:
                    continue
            image_bytes = (Path(self.ls.store.out) / group[0]["image_path"]).read_bytes()  # snapshot now
            try:
                obs, jpeg = build_observation(group, image_bytes, self.pipeline)
            except NoCaptureTime:
                self.stats["skipped_no_capture_time"] += 1
                continue
            self.stats["below_contract_min_confidence"] += len(below_contract_threshold(obs))
            self.stats["observations"] += 1
            for r in group:
                self.sent_ids[r["memory_id"]] = obs["observation_id"]
            if self.uploader:
                self.uploader.enqueue(obs, jpeg)
            if self.outbox:
                self._write_outbox(obs, jpeg)
            ids.append(obs["observation_id"])
        return ids

    def found(self, label: str, found_at_ms: int | None = None):
        """User said "found it": delete every local memory of that object, then queue the found action.

        found_at is taken first, from this machine's clock (a live user action, not footage time).
        The uploader then (under its lock) moves queued sightings of only this label captured at or
        before found_at to superseded/, and queues POST /items/{label}/found behind everything else,
        so a backlog can't reopen the item after it. With the uploader's background thread running,
        this returns at once and the thread sends it (with the usual retries); without one, it
        flushes the queue here, in order. Returns removed_locally, found_at, observation_id (last
        one sent for these memories), found_id, superseded (observation ids), queued (True while the
        found action is still waiting to be sent) and backend_status (the found endpoint's last
        HTTP status; None if not tried yet or with no uploader; 404 = no sighting server-side).
        """
        found_ms = found_at_ms if found_at_ms is not None else int(time.time() * 1000)
        label = item_label(label)
        # every prompt that maps to this item ("bunch of keys", "keychain", ...): ALL its memories go
        prompts = list(dict.fromkeys(r["label"] for r in self.ls.memories() if item_label(r["label"]) == label))
        last_obs, removed = None, []
        for prompt in prompts or [label]:
            for r in self.ls.memories(prompt):
                last_obs = last_obs or self.sent_ids.get(r["memory_id"])
                self.selector.forget(r["memory_id"])
            removed += self.ls.found(label=prompt)
        found_at = _iso(found_ms)
        result = {"removed_locally": removed, "found_at": found_at, "observation_id": last_obs,
                  "found_id": None, "superseded": [], "queued": False, "backend_status": None}
        if self.uploader:
            queued = self.uploader.queue_found(label, found_at)
            if not self.uploader.running:
                self.uploader.flush()
            result.update(found_id=queued["id"], superseded=queued["superseded"],
                          queued=self.uploader.is_pending(queued["id"]),
                          backend_status=self.uploader.http_status.get(queued["id"]))
        return result

    def _write_outbox(self, obs, jpeg):
        stem = self.outbox / obs["observation_id"]
        stem.with_suffix(".json").write_text(json.dumps(obs, indent=2), encoding="utf-8")
        stem.with_suffix(".jpg").write_bytes(jpeg)
        preview = cv2.imdecode(__import__("numpy").frombuffer(jpeg, "uint8"), cv2.IMREAD_COLOR)
        w, h = obs["image_size"]["width"], obs["image_size"]["height"]
        for o in obs["objects"]:  # draw the box FROM the normalized values, to prove the conversion
            x, y, bw, bh = o["bbox"]
            cv2.rectangle(preview, (int(x * w), int(y * h)), (int((x + bw) * w), int((y + bh) * h)), (0, 0, 255), 3)
            cv2.putText(preview, f"{o['label']} {o['confidence']:.2f} ({o['event']})", (int(x * w), max(24, int(y * h) - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imwrite(str(stem) + "_preview.jpg", preview)


def _iso(ms):
    return datetime.datetime.fromtimestamp(ms / 1000, tz=datetime.timezone.utc).isoformat(
        timespec="milliseconds").replace("+00:00", "Z")
