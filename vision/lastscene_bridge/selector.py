"""Which memory events become observations ("send events, not frames").

Our memory engine emits: created, image_updated, resighted, sighting_ended (per memory).
Policy, per memory:
  created                          -> send (first confirmed sighting of this object here)
  image_updated at sighting end    -> send only if the evidence moved noticeably since the last
                                      sent observation (bbox center shifted > move_threshold of the
                                      frame) or more than 10 minutes passed (contract: "seen again
                                      after more than 10 minutes")
  image_updated mid-sighting       -> skip (just a clearer frame)
  resighted                        -> skip (carries the OLD evidence frame; a fresh frame follows at
                                      sighting end and is judged by the rule above)
  sighting_ended                   -> skip
Events from the same evidence frame are grouped so they become one observation (contract: batch).
"""

from collections import OrderedDict

RESEND_AFTER_MS = 10 * 60 * 1000


def _center_norm(record):
    x1, y1, x2, y2 = record["bbox"]
    w, h = record.get("_image_size", (None, None))
    if not w:
        return None
    return ((x1 + x2) / 2 / w, (y1 + y2) / 2 / h)


def _time_ms(record):
    return record["observed_ms"] if record.get("observed_ms") is not None else record["video_ms"]


class EventSelector:
    def __init__(self, move_threshold: float = 0.10, resend_after_ms: int = RESEND_AFTER_MS):
        self.move_threshold = move_threshold
        self.resend_after_ms = resend_after_ms
        self.sent = {}  # memory_id -> (time_ms, normalized center or None)

    def select(self, events: list[dict]) -> list[list[dict]]:
        """events: one batch as returned by process_frame()/finish(). Returns groups (one per evidence frame)."""
        chosen = []
        for i, e in enumerate(events):
            kind, mid = e["event"], e["memory_id"]
            if kind == "created":
                chosen.append(e)
            elif kind == "image_updated" and self._ends_sighting(events, i):
                if self._worth_resending(e):
                    chosen.append(e)
        groups = OrderedDict()
        for e in chosen:
            self.sent[e["memory_id"]] = (_time_ms(e), _center_norm(e))
            groups.setdefault((e.get("source"), e.get("observed_ms"), e.get("video_ms")), []).append(e)
        return list(groups.values())

    def forget(self, memory_id: str):
        self.sent.pop(memory_id, None)

    @staticmethod
    def _ends_sighting(events, i):
        mid = events[i]["memory_id"]
        return any(e["event"] == "sighting_ended" and e["memory_id"] == mid for e in events[i + 1:])

    def _worth_resending(self, e):
        last = self.sent.get(e["memory_id"])
        if last is None:
            return True
        last_t, last_c = last
        if _time_ms(e) - last_t > self.resend_after_ms:
            return True
        c = _center_norm(e)
        if c is None or last_c is None:
            return False
        return ((c[0] - last_c[0]) ** 2 + (c[1] - last_c[1]) ** 2) ** 0.5 > self.move_threshold
