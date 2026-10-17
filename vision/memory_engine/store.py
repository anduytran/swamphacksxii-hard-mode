"""Write memories to disk in a form Dylan's backend can read without our code.

Output folder layout:
    <out>/memories.json          current state of every memory (rewritten on each change)
    <out>/events.jsonl           append-only log: one JSON event per line
    <out>/images/<memory_id>.jpg       full scene (clean, no drawings; bbox is in the JSON)
    <out>/images/<memory_id>_crop.jpg  close-up of the object

Image paths in the JSON are relative to <out>, so the folder can be moved or served as-is.

Memories from earlier runs in the same folder are kept (loaded on start), so a restart
doesn't forget where the keys were. delete() removes everything about one memory:
both images, its record, and its lines in events.jsonl (replaced by a small tombstone).
"""

import json
import os
import time
from pathlib import Path

import cv2

SAVES_IMAGE = {"created", "image_updated"}


class MemoryStore:
    def __init__(self, out_dir: str, source: str, session_id: str = "", source_kind: str = "",
                 keep_existing: bool = True):
        self.out = Path(out_dir)
        (self.out / "images").mkdir(parents=True, exist_ok=True)
        self.source = source
        self.session_id = session_id
        self.source_kind = source_kind
        self.records: dict[str, dict] = {}
        snapshot = self.out / "memories.json"
        if keep_existing and snapshot.exists():
            for record in json.loads(snapshot.read_text(encoding="utf-8")).get("memories", []):
                self.records[record["memory_id"]] = record
        self._events = (self.out / "events.jsonl").open("a", encoding="utf-8")

    def delete(self, memory_id: str) -> dict | None:
        """Remove a memory completely. Returns what was removed, or None if it didn't exist."""
        record = self.records.pop(memory_id, None)
        if record is None:
            return None
        removed = []
        for key in ("image_path", "crop_path"):
            if record.get(key):
                path = self.out / record[key]
                if path.exists():
                    path.unlink()
                    removed.append(record[key])
        # scrub this memory's history (bboxes, times, paths) from the event log
        self._events.close()
        log = self.out / "events.jsonl"
        kept = [line for line in log.read_text(encoding="utf-8").splitlines()
                if line.strip() and json.loads(line).get("memory_id") != memory_id]
        tombstone = {"event": "deleted", "memory_id": memory_id, "label": record.get("label"),
                     "wall_ms": int(time.time() * 1000)}
        log.write_text("".join(line + "\n" for line in kept) + json.dumps(tombstone) + "\n", encoding="utf-8")
        self._events = log.open("a", encoding="utf-8")
        self._write_snapshot()
        return {"memory_id": memory_id, "label": record.get("label"), "files_removed": removed}

    def handle(self, event: dict, image=None) -> dict:
        """Save images if needed, log the event, refresh memories.json. Returns the event with paths.

        The image saved is the one attached to the event by MemoryMaker (the frame the
        evidence actually came from); `image` is only a fallback for callers that don't attach one.
        """
        memory_id = event["memory_id"]
        attached = event.pop("_image", None)
        image = attached if attached is not None else image
        if event["event"] in SAVES_IMAGE:
            event["image_path"] = f"images/{memory_id}.jpg"
            event["crop_path"] = f"images/{memory_id}_crop.jpg"
            cv2.imwrite(str(self.out / event["image_path"]), image)
            cv2.imwrite(str(self.out / event["crop_path"]), crop(image, event["bbox"]))
        else:
            previous = self.records.get(memory_id, {})
            event["image_path"] = previous.get("image_path")
            event["crop_path"] = previous.get("crop_path")
        event["source"] = self.source
        event["session_id"] = self.session_id
        event["source_kind"] = self.source_kind

        self._events.write(json.dumps(event) + "\n")
        self._events.flush()
        self.records[memory_id] = {k: v for k, v in event.items() if k != "event"}
        self._write_snapshot()
        return event

    def close(self):
        self._events.close()

    def _write_snapshot(self):
        snapshot = {"source": self.source, "session_id": self.session_id, "source_kind": self.source_kind,
                    "memories": list(self.records.values())}
        tmp = self.out / "memories.json.tmp"
        tmp.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
        os.replace(tmp, self.out / "memories.json")  # atomic: readers never see a half-written file


def crop(image, bbox, pad=0.5):
    """Object close-up with `pad` (as a fraction of box size) of surroundings on each side."""
    h, w = image.shape[:2]
    x1, y1, x2, y2 = bbox
    px, py = int((x2 - x1) * pad), int((y2 - y1) * pad)
    return image[max(0, y1 - py):min(h, y2 + py), max(0, x1 - px):min(w, x2 + px)]
