"""Live demo loop: the bridge's frame loop plus "where are my keys?" guidance and "found it".

    python -m lastscene_bridge.run --source http://<cam-ip>:81/stream --show

Commands, as keys in the window or as lines typed into the console (console works without
window focus and in headless runs):
    w / where    print the latest memory (when, which photo) and start guidance
    f / found    "found it": bridge.found("keys") deletes the local memories and tells the backend
    q / quit     stop; run.py then flushes and stops the uploader as usual

What is drawn (and printed) stays honest:
- "visible now" only ever labels a box the detector returned for THIS frame.
- guide()'s projected box (the remembered spot mapped into the live view by scene matching)
  is dashed and labelled "memory: last seen here". It stays local and is never called visible.
- The memory's own box is drawn on its evidence photo (the inset), never on the live frame.

Guidance costs no extra inference: FrameDetections hands guide()'s locate() the detections
that process_frame already computed for the same frame.
"""

import datetime
import json
import os
import queue
import sys
import threading
import time

import cv2

from lastscene_bridge.observation import item_label
from memory_engine.api import resolve_label

WINDOW = "LastSeen live (w: where are my keys  f: found it  q: quit)"
COMMANDS = {"w": "where", "where": "where", "where are my keys": "where",
            "f": "found", "found": "found", "found it": "found",
            "q": "quit", "quit": "quit", "exit": "quit"}
KEYS = {ord("w"): "where", ord("f"): "found", ord("q"): "quit", 27: "quit"}  # 27 = Esc

LIVE = (0, 200, 0)        # BGR: detector box in this frame
MEMORY = (0, 140, 255)    # projected "last seen here" box
TEXT = (255, 255, 255)


def parse_command(text: str) -> str | None:
    return COMMANDS.get(text.strip().lower().rstrip("?!."))


def key_command(key: int) -> str | None:
    return KEYS.get(key & 0xFF) if key >= 0 else None


class FrameDetections:
    """Detector wrapper that reuses the last result when asked about the same frame again.

    process_frame and guide() -> locate() both call detector.detect(image) on the same array;
    with this they share one inference (2/s instead of 4/s while guiding).
    """

    def __init__(self, inner):
        self.inner = inner
        self._image = None
        self._dets = []

    def detect(self, image):
        if image is not self._image:  # identity, not equality: we hold the array, so it can't be reused
            self._dets = list(self.inner.detect(image))
            self._image = image
        return list(self._dets)

    def __getattr__(self, name):
        return getattr(self.__dict__["inner"], name)


def _clock(ms: int | None) -> str:
    return datetime.datetime.fromtimestamp(ms / 1000).strftime("%H:%M:%S") if ms else "?"


def _ago(ms: int | None, now_ms: int) -> str:
    if not ms:
        return "unknown time"
    s = max(0, (now_ms - ms) // 1000)
    return f"{s}s ago" if s < 120 else f"{s // 60} min ago" if s < 7200 else f"{s // 3600} h ago"


def describe_memory(mem: dict, now_ms: int) -> list[str]:
    """Console answer to "where are my keys?" from latest_memory()."""
    when = mem.get("last_seen_observed_ms") or mem.get("last_seen_wall_ms")
    basis = "capture time" if mem.get("time_basis") == "observed" else "processing time (capture time unknown)"
    lines = [f"Last seen {_clock(when)} ({_ago(when, now_ms)}; {basis}), seen {mem.get('sightings', 1)}x",
             f"  place: {mem.get('place_status')}   detector confidence {mem.get('confidence', 0):.2f}",
             f"  evidence photo: {mem.get('image_file')}   box {mem.get('bbox')}"]
    lines += [f"  note: {c}" for c in mem.get("caveats", [])]
    return lines


def _where_words(d: dict | None) -> str:
    if not d:
        return ""
    h = {"left": "left", "right": "right"}.get(d.get("horizontal"))
    v = {"up": "up", "down": "down"}.get(d.get("vertical"))
    words = " and ".join(w for w in (v, h) if w)
    if not d.get("in_frame"):
        return f"turn {words or 'around'}"
    return f"look {words}" if words else "straight ahead"


def guidance_cue(g: dict, item: str = "keys") -> str:
    """One line of guidance from guide(). Says "visible now" only with a live detection."""
    status = g.get("status")
    visible = bool((g.get("detection") or {}).get("visible"))
    where = _where_words(g.get("direction"))
    if visible and status == "object_at_remembered_spot":
        return f"{item.upper()} VISIBLE NOW at the remembered spot: {where}"
    if visible and status == "object_seen_elsewhere":
        return f"{item.upper()} VISIBLE NOW, not where the memory says: {where}"
    if visible:
        return f"{item.upper()} VISIBLE NOW: {where}"
    if status == "remembered_spot_in_view":
        return f"Not detected right now. Memory: last seen here, {where}"
    if status == "remembered_spot_out_of_frame":
        return f"Memory: last-seen spot is out of view, {where}"
    if status == "remembered_place_not_recognized":
        return "Place not recognized yet: look around for the place in the memory photo"
    if status == "no_such_memory":
        return "That memory no longer exists"
    return f"No guidance ({status})"


def guidance_overlay(g: dict, shape) -> list[dict]:
    """Boxes/arrow to draw for one guide() result, in live-frame pixels."""
    items = []
    if g.get("remembered_box_in_view"):
        items.append({"kind": "memory", "box": g["remembered_box_in_view"], "style": "dashed",
                      "label": "memory: last seen here"})
    det = g.get("detection") or {}
    if det.get("visible") and det.get("bbox"):
        items.append({"kind": "live", "box": det["bbox"], "style": "solid",
                      "label": f"visible now {det.get('confidence', 0):.2f}"})
    if g.get("target_box") and g.get("direction"):
        h, w = shape[:2]
        x1, y1, x2, y2 = g["target_box"]
        tx = min(max((x1 + x2) / 2, 20), w - 20)  # clip to the frame so off-screen targets point at the edge
        ty = min(max((y1 + y2) / 2, 20), h - 20)
        if g["direction"]["horizontal"] != "center" or g["direction"]["vertical"] != "middle":
            items.append({"kind": "arrow", "from": (w // 2, h // 2), "to": (int(tx), int(ty))})
    return items


def detection_overlay(dets) -> list[dict]:
    return [{"kind": "live", "box": list(d.bbox), "style": "solid", "label": f"visible now {d.conf:.2f}"}
            for d in dets]


def _dashed_rect(img, box, color, thickness=3, dash=14):
    x1, y1, x2, y2 = (int(v) for v in box)
    for (ax, ay), (bx, by) in (((x1, y1), (x2, y1)), ((x2, y1), (x2, y2)), ((x2, y2), (x1, y2)), ((x1, y2), (x1, y1))):
        length = max(abs(bx - ax), abs(by - ay))
        for s in range(0, length, 2 * dash):
            e = min(s + dash, length)
            p = (ax + (bx - ax) * s // max(length, 1), ay + (by - ay) * s // max(length, 1))
            q = (ax + (bx - ax) * e // max(length, 1), ay + (by - ay) * e // max(length, 1))
            cv2.line(img, p, q, color, thickness)


def _text(img, text, org, scale=0.7, color=TEXT):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)


def draw(img, items):
    for it in items:
        if it["kind"] == "arrow":
            cv2.arrowedLine(img, it["from"], it["to"], MEMORY, 6, tipLength=0.08)
            continue
        color = LIVE if it["kind"] == "live" else MEMORY
        x1, y1, x2, y2 = (int(v) for v in it["box"])
        if it["style"] == "dashed":
            _dashed_rect(img, it["box"], color)
        else:
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        _text(img, it["label"], (x1, max(22, y1 - 8)), 0.6, color)
    return img


class LiveSession:
    """Runs frames through LastSeen + bridge and handles where / found / quit between frames.

    All LastSeen calls happen on the frame-loop thread; the console thread only queues commands.
    """

    def __init__(self, ls, bridge, item: str | None = None, show: bool = False, uploading: bool = False,
                 log=None, now_ms=None):
        self.ls, self.bridge = ls, bridge
        self.prompt = ls.prompts[0] if item is None else resolve_label(item)
        self.item = item_label(self.prompt)  # contract label for found(): "bunch of keys" -> "keys"
        self.show, self.uploading = show, uploading
        self.log = log or (lambda msg: print(msg, flush=True))
        self.now_ms = now_ms or (lambda: int(time.time() * 1000))
        self.commands = queue.Queue()
        self.guiding = False
        self.memory_id = None
        self.last_guide = None
        self.notice, self.notice_until = None, 0
        self.stopped = False
        self._inset = None
        self._window_open = False

    # --- commands ------------------------------------------------------------
    def submit(self, command: str):
        self.commands.put(command)

    def start_console(self, lines=None):
        """Read where / found / quit lines on a daemon thread (default: this process's stdin)."""
        if lines is None:
            try:
                fd = sys.stdin.fileno()
            except (AttributeError, OSError, ValueError):  # no stdin (pythonw) or one without a real fd
                return None
            lines = _fd_lines(fd)

        def read():
            for line in lines:
                cmd = parse_command(line)
                if cmd:
                    self.submit(cmd)
                elif line.strip():
                    self.log(f"unknown command {line.strip()!r}: type where, found or quit")

        t = threading.Thread(target=read, name="console-commands", daemon=True)
        t.start()
        return t

    def handle_commands(self):
        while not self.commands.empty():
            cmd = self.commands.get_nowait()
            if cmd == "where":
                self.where()
            elif cmd == "found":
                self.found()
            elif cmd == "quit":
                self.log("quit requested")
                self.stopped = True

    def where(self):
        mem = self.ls.latest_memory(self.prompt)
        self.log(f"> where are my {self.item}?")
        if mem is None:
            self.log(f"  No memory of {self.item} yet: put them down in view of the camera.")
            self._notify(f"No memory of {self.item} yet")
            return
        for line in describe_memory(mem, self.now_ms()):
            self.log("  " + line)
        self.guiding, self.memory_id = True, mem["memory_id"]
        self._inset = None
        self.log("  guidance on (f = found it)")

    def found(self):
        self.log(f"> found it ({self.item})")
        self.guiding, self.memory_id, self.last_guide = False, None, None
        try:
            result = self.bridge.found(self.item)
        except Exception as exc:  # keep the demo loop alive; the error is shown, not hidden
            self.log(f"  found failed: {type(exc).__name__}: {exc}")
            self._notify(f"Found it: backend call failed ({type(exc).__name__})")
            return None
        self.log("  " + json.dumps(result, default=str))
        removed = len(result.get("removed_locally") or []) if isinstance(result, dict) else 0
        status = result.get("backend_status") if isinstance(result, dict) else None
        self._notify(f"Found it: {removed} local memor{'y' if removed == 1 else 'ies'} cleared"
                     + (f", backend {status}" if status is not None else ", not sent (dry run)" if not self.uploading else ""))
        return result

    # --- frame loop ----------------------------------------------------------
    def run(self, frames):
        for f in frames:
            self.step(f)
            if self.stopped:
                break
        if not self.stopped:
            self.handle_commands()  # e.g. "found" typed while a dropped stream was reconnecting

    def step(self, f) -> list[str]:
        self.handle_commands()
        if self.stopped:
            return []
        dets = self.ls.detector.detect(f.image)  # the one inference for this frame (see FrameDetections)
        events = self.ls.process_frame(f.image, f.video_ms, f.wall_ms, f.observed_ms, detections=dets)
        ids = self.bridge.handle(events)
        for e in events:
            if e["event"] == "created":
                self.log(f"[{f.video_ms / 1000:6.1f}s] memory {e['memory_id']} ({e['label']}, "
                         f"confidence {e['confidence']:.2f})")
        for oid in ids:
            self.log(f"[{f.video_ms / 1000:6.1f}s] observation {oid} {'queued' if self.uploading else 'written'}")
        g = self._guide(f) if self.guiding else None
        if self.show:
            self._render(f, dets, g)
        return ids

    def _guide(self, f):
        mem = self.ls.latest_memory(self.prompt)
        if mem is None:
            self.log(f"  guidance off: no memory of {self.item} left")
            self.guiding, self.memory_id = False, None
            return None
        if mem["memory_id"] != self.memory_id:
            self.log(f"  newer memory {mem['memory_id']}, guiding to it")
            self.memory_id, self._inset = mem["memory_id"], None
        g = self.ls.guide(f.image, self.memory_id)
        self.last_guide = g
        self.log(f"[{f.video_ms / 1000:6.1f}s] guide: {guidance_cue(g, self.item)}  "
                 f"(status {g.get('status')}, scene {g.get('scene_verdict')})")
        return g

    def _notify(self, text, seconds=4.0):
        self.notice, self.notice_until = text, time.monotonic() + seconds

    # --- window --------------------------------------------------------------
    def _render(self, f, dets, g):
        vis = f.image.copy()
        h, w = vis.shape[:2]
        scale = max(0.6, w / 1600)
        draw(vis, guidance_overlay(g, vis.shape) if g else
             detection_overlay([d for d in dets if d.label in self.ls.prompts]))
        if g:
            _text(vis, guidance_cue(g, self.item), (16, h - 60), scale)
            self._draw_inset(vis)
        mems = self.ls.memories(self.prompt)
        latest = mems[0] if mems else None
        when = latest and (latest.get("last_seen_observed_ms") or latest.get("last_seen_wall_ms"))
        status = (f"{'GUIDING' if g else 'watching'} | {len(mems)} memor{'y' if len(mems) == 1 else 'ies'} of "
                  f"{self.item}" + (f", latest {_clock(when)}" if latest else "")
                  + f" | observations {self.bridge.stats['observations']} {'queued' if self.uploading else 'written (dry run)'}")
        _text(vis, status, (16, int(34 * scale)), scale)
        _text(vis, "w: where are my keys   f: found it   q: quit", (16, h - 20), scale * 0.8)
        if self.notice and time.monotonic() < self.notice_until:
            _text(vis, self.notice, (16, int(80 * scale)), scale * 1.2, (0, 255, 255))
        cv2.imshow(WINDOW, vis)
        self._window_open = True
        cmd = key_command(cv2.waitKey(1))
        if cmd:
            self.submit(cmd)
        elif cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            self.submit("quit")  # window closed with the mouse

    def _draw_inset(self, vis):
        """The memory's evidence photo with its own (real detector) box, top right."""
        if self._inset is None:
            record = self.ls.store.records.get(self.memory_id) or {}
            img = cv2.imread(str(self.ls.store.out / record["image_path"])) if record.get("image_path") else None
            if img is None:
                return
            x1, y1, x2, y2 = (int(v) for v in record["bbox"])
            cv2.rectangle(img, (x1, y1), (x2, y2), MEMORY, max(3, img.shape[1] // 300))
            tw = vis.shape[1] // 4
            img = cv2.resize(img, (tw, int(img.shape[0] * tw / img.shape[1])))
            when = record.get("last_seen_observed_ms") or record.get("last_seen_wall_ms")
            _text(img, f"memory photo {_clock(when)}", (8, 24), 0.6, MEMORY)
            self._inset = img
        ih, iw = self._inset.shape[:2]
        if ih < vis.shape[0] and iw < vis.shape[1]:
            vis[8:8 + ih, -iw - 8:-8] = self._inset

    def close(self):
        if self._window_open:
            cv2.destroyAllWindows()
            cv2.waitKey(1)


def _fd_lines(fd):
    """Lines from a file descriptor via os.read: unbuffered, so a daemon thread blocked here
    holds no lock on sys.stdin at interpreter shutdown."""
    buf = b""
    while True:
        try:
            chunk = os.read(fd, 1024)
        except OSError:
            return
        if not chunk:
            if buf:
                yield buf.decode("utf-8", "replace")
            return
        buf += chunk
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            yield line.decode("utf-8", "replace")
