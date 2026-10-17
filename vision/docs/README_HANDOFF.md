# LastSeen ML: integration handoff for Dylan

From Adam (ML/data science), Oct 17, 2026. This is the ML side of LastSeen as it exists today:
- turning camera frames into object memories;
- answering "where are my keys?";
- guidance while you look around;
- deleting the memory on "found it".

**How it connects to LastScene (per your Oct 17 contract):** the pipeline runs on its own host (a laptop, with the camera streaming to it). It **POSTs observations** (full JPEG + JSON) to `/api/v1/observations`, and runs `guide()` / `locate()` locally.
- **Start with `LASTSCENE_INTEGRATION.md`:** the field-by-field audit, an example payload, uploader behavior, decisions we need from you, and what was verified.
- **This file** documents the ML package itself (`memory_engine`).

> **Status in one line:** the observation format, uploader and ML API are tested against a local mock written from the contract. Your real endpoints aren't tested yet. Detection *accuracy* depends on the preset (`m960d` for now; final pick after ESP32 footage).

## 1. What's in this folder

```
LASTSCENE_INTEGRATION.md   READ FIRST: contract audit, example payload, uploader, decisions for Dylan
README_HANDOFF.md          this file: the ML package, its API and its local data format
requirements.txt           pinned versions (+ the PyTorch CUDA install line)
.env.example               names of the credential environment variables (no values)
pytest.ini
memory_engine/             the ML package (unchanged by the integration)
  api.py                   LastSeen class: process_frame / latest_memory / guide / locate / found
  presets.py               detector presets (model, input size, distractor labels)
  memory.py, scene.py, store.py, detector.py, frames.py, run.py, gpu_monitor.py
lastscene_bridge/          contract side: memory events -> observations -> uploader
  observation.py           JSON + JPEG per the contract (normalized bbox, UUIDs, UTC captured_at)
  selector.py              which events are sent ("events, not frames")
  uploader.py, auth.py     Auth0 token, X-Pairing-Code, rate limit, retries, offline queue
  capture_time.py          real recording time from video metadata (no invented timestamps)
  bridge.py, run.py        glue + CLI: python -m lastscene_bridge.run --source ... [--dry-run]
examples/quickstart.py     runs every ML API call (works without a GPU: --no-model)
samples/
  lastscene_observation/   REAL observation payload + the exact uploaded JPEG + preview (box drawn from normalized bbox)
  file_run/                REAL output of run.py on a phone clip: memories.json, events.jsonl, images/
  live_api_run/            REAL API output used like a live loop + api_outputs.json (latest_memory/guide/locate/found)
eval/
  manifests/               labeled test clips (phone_round2) + the ESP32 template (esp32_round1)
  summary_phone_round2.md  latest benchmark across 5 presets
  reports/<preset>/        full text reports + confidence timelines (charts only, no footage)
scripts/                   eval_run.py, record_stream.py, contact_sheet.py, compare_frames.py, detect_test.py
tests/                     100 tests (+ 8 real-clip tests that skip without the videos)
  mock_lastscene.py        local mock of your API + Auth0, written from the contract (runnable by hand)
docs/PROJECT_NOTES.md      Adam's full README: every benchmark and decision so far
```

**Not included, on purpose:**
- the virtualenv;
- model weights (they download on first run);
- any `.env` or credential values (the uploader reads environment variables; see `.env.example`);
- caches, queues and memory stores;
- the raw videos.

## 2. Setup and runtime requirements

- **Python 3.12**, Windows or Linux.
- **An NVIDIA GPU is strongly recommended.** Tested on an RTX 4060 Laptop with driver 592. It runs on the CPU too, but slowly.
- **Install:**
  ```
  python -m venv .venv
  .venv\Scripts\Activate.ps1            # Linux/macOS: source .venv/bin/activate
  pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
  pip install -r requirements.txt
  pytest -q                              # expect: 100 passed, 8 skipped
  ```
- **First detector use needs internet.** Ultralytics downloads, into the *current working directory*:
  - `yolov8m-worldv2.pt` (57 MB);
  - `weights/clip/ViT-B-32.pt` (354 MB);
  - `yolov8x-worldv2.pt` (146 MB) only when a search/XL preset is used.

  Start your backend from a fixed directory, or copy those files there beforehand. The download takes about 30–60 s. Loading the model takes about 3 s.
- **Licenses:** YOLO-World via Ultralytics is AGPL-3.0, fine for our public repo but keep the credit. OpenCV is Apache-2.0. Built with help from Claude Code (Anthropic); credit AI tools in the Devpost, per the rules.

## 3. How to call it

### Initialization (once, at startup)

```python
from memory_engine.api import LastSeen, resolve_label

ls = LastSeen(
    out_dir="data/memories_out",   # where memories.json, events.jsonl and images/ live (persists across restarts)
    preset="efficient",            # current default; see section 6 for when to use "m960d"
    scene_matching=True,           # default; decides "same place" from the surroundings
    source="esp32-cam-1",          # free text, stored on each memory
    source_kind="live",            # "live" for a camera, "file" for a recording
)
```

- **Models load lazily.** The detector loads on the first `process_frame` / `locate` / `guide` call. The XL search model loads on the first `locate(..., search=True)` (+146 MB, about 3 s).
- **Not thread-safe.** Make all calls from **one worker thread**, e.g. a `ThreadPoolExecutor(max_workers=1)` that owns the instance. The camera loop and any local UI can then share it safely. (`lastscene_bridge.run` does exactly this in a single loop.)
- **Frames are OpenCV BGR `numpy.uint8` arrays**, shape `(H, W, 3)`.
  - From JPEG bytes: `cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)`.
  - From RGB: `frame[:, :, ::-1]`.

### `process_frame(frame, video_ms, wall_ms=None, observed_ms=None)` → list of events

Call this from the camera loop **2 times per second**. It's the rate everything was tuned and soak-tested at: the GPU stays near idle, about 5–14 W at 50–55 °C.
- `video_ms`: ms since your camera session started. Must increase monotonically.
- For a live camera, pass `observed_ms=int(time.time()*1000)`; `wall_ms` defaults to now.
- It returns the events it just wrote (`created`, `image_updated`, `resighted`, `sighting_ended`). You can ignore them and read `memories.json`, or push them to the dashboard.
- Cost: about 15–20 ms per call on the 4060 with `efficient` or `m960d`; about 30 ms with `x960d`.

### `latest_memory(label)` → dict or `None`  ("where are my keys?")

```python
mem = ls.latest_memory(resolve_label("my keys"))   # resolve_label: "keys"/"my keys"/"car keys" -> "bunch of keys"
```

- Returns the **most recent** sighting of that object: recency beats a clearer older photo.
- Returns `None` if there's no memory.
- The result is the stored record (section 4) **plus** these API-only fields:
  - `image_file`, `crop_file`: paths to the evidence images (`out_dir` + `image_path`);
  - `time_basis`: `"observed"` (real-world time) or `"processing"` (only file runs without a recording time);
  - `caveats`: sentences to show the user, e.g. "Couldn't confirm whether this is a new spot…";
  - `alternatives`: up to two older memory ids for the same object.

Real example: `samples/live_api_run/api_outputs.json`, first entry.

### `guide(frame, memory_id)` → dict  (while the user looks around)

Call it on live frames, 2–5 times per second, **only during an active search**. It compares the live view with the memory's evidence photo, and reports a live detection alongside.

| `status` | Meaning | What the UI should say |
|---|---|---|
| `object_at_remembered_spot` | Detector sees the keys where they were remembered | "There they are" + draw `target_box` |
| `object_detected` | Detector sees keys; remembered place not recognized | "I see keys here" + box |
| `object_seen_elsewhere` | Place recognized, but keys detected somewhere else in view | "They may have moved; I see keys here" |
| `remembered_spot_in_view` | Place recognized, the remembered spot is in view, **keys not detected** | "They were here" + box. Never claim they're there now: it may be a detector miss, or they were moved |
| `remembered_spot_out_of_frame` | Place recognized, spot is off-screen | arrow from `direction` |
| `remembered_place_not_recognized` | Can't match the view, no detection | show the memory photo: "go to the area in this photo" |

Other fields:
- `target_box`: draw this.
- `direction`:
  - `horizontal` is left/center/right; `vertical` is up/middle/down;
  - `offset_x` and `offset_y` run from -1 to 1 within the frame;
  - `in_frame` is false when the target is off-screen.
- `remembered_box_in_view`, `visible_fraction`, `scene_verdict`, `scene_inliers`, `projection_trusted`.
- `detection`: the raw `locate()` result.

**Rule we verified on every evaluated frame: "keys visible" is only ever claimed when `detection.visible` is true.**

Real examples at 6.0 s and 10.8 s of the return clip: `api_outputs.json`, entries 2–3.

### `locate(frame, label, search=False)` → dict

A live detection only: `{"visible": bool, "bbox", "confidence", "offset_x", "offset_y", "horizontal", "vertical", "in_frame", "model"}`.
- `search=True` runs the XL model on the whole frame plus 2×2 tiles. It finds smaller and farther keys, but costs about 5× an XL call, so use it only when the user asks to search.

### `found(label=...)` → list  ("found it")

```python
ls.found(label=mem["label"])     # use the label form, not found(memory_id)
bridge.found("keys")             # with the LastScene bridge: the same local deletion + POST /api/v1/items/keys/found
```

- **Removes:** both images, the record from `memories.json`, that memory's lines in `events.jsonl` (a short `deleted` tombstone stays), and cached scene data.
- **Pauses new memories of that label for 5 s,** so the keys in your hand don't immediately become a new memory.
- **Why the label form:** when scene matching can't confirm "same spot from another angle", the same keys can have an older `unverified` memory. That older one would otherwise resurface as the next answer; it happened in our eval with `x960d`.
- **Returns:** `[{"memory_id", "label", "files_removed": [...]}]`.

### Also available

- `ls.memories(label=None)`: all records, newest first.
- `ls.finish()`: closes open sightings when a session ends.
- `ls.close()`: closes the event log.

## 4. Data contract (exact current schema)

Files in `out_dir`:
- `memories.json`: the current state, rewritten atomically after every change, so you never read a half-written file.
- `events.jsonl`: an append-only log, one event per line.
- `images/<memory_id>.jpg`: the full camera frame, clean.
- `images/<memory_id>_crop.jpg`: a close-up with 50% padding.

### `memories.json` top level

| Field | Meaning |
|---|---|
| `source`, `session_id`, `source_kind` | of the *current* session |
| `memories` | list of records. Memories from earlier sessions in the same folder are kept, and each record carries its own `session_id`. |

### Memory record fields

**Times:**
- `*_video_ms` are **relative**: ms since that session/source started. Only comparable within one `session_id`.
- `*_wall_ms` and `*_observed_ms` are **absolute** Unix epoch milliseconds (UTC).

| Field | Type | Meaning |
|---|---|---|
| `memory_id` | str | `mem_<label-slug>_<8 hex>`; unique, stable for the life of the memory |
| `label` | str | detector prompt, e.g. `"bunch of keys"` (use `resolve_label()` for user words) |
| `bbox` | [x1, y1, x2, y2] int | object box in **pixels of the saved image** (= the camera frame size; read width/height from the image) |
| `confidence` | float | detector confidence of the evidence frame (typical range 0.05–0.5; low numbers are normal for this model) |
| `frame_id` | int | index of the processed frame within the session (not a camera frame number) |
| `video_ms` | int, relative | when the **evidence image** was taken |
| `wall_ms` | int, absolute | when the evidence frame was processed |
| `observed_ms` | int or null, absolute | when the evidence frame was **seen in the real world**. For live sources it equals `wall_ms`. For files it's `--recorded-at` + `video_ms`, or **null** if no recording time was given |
| `first_seen_video_ms` | int, relative | first frame of the sighting that created this memory |
| `last_seen_video_ms` | int, relative | latest frame in which the object was detected (can be later than `video_ms`) |
| `last_seen_wall_ms` | int, absolute | processing time of that latest detection |
| `last_seen_observed_ms` | int or null, absolute | real-world time of that latest detection (null as above) |
| `sightings` | int | separate times this object was seen at this place (resightings increase it) |
| `place_status` | str or null | how this memory relates to earlier ones (null only with scene matching off). One of: `first_sighting`, `likely_new_place` (every comparison said different place; evidence, not proof), `moved_in_scene`, `unverified` (couldn't tell), `another_instance` (two of the same label in view at once) |
| `scene_verdict` | str or null | `same_scene` / `different_scene` / `unknown` from the comparison that decided `place_status` |
| `scene_inliers` | int or null | matched feature count for that comparison (more = more confident; about 25+ = match) |
| `compared_with` | str or null | `memory_id` it was compared against |
| `image_path`, `crop_path` | str | relative to `out_dir` (serve that folder as static files) |
| `source`, `session_id`, `source_kind` | str | where the memory came from (`source_kind`: `live` or `file`) |

**For ordering and "how long ago":**
- Use `last_seen_observed_ms` when it's not null, otherwise `last_seen_wall_ms`. That's exactly what `latest_memory()` does.
- Use `video_ms` only to line things up within one session.

### `events.jsonl`

Each line is a record snapshot plus `"event"`:

| `event` | Meaning |
|---|---|
| `created` | new memory, images written |
| `image_updated` | a better or more recent evidence frame replaced the images. A sighting normally ends with one, so the photo shows the *latest* confident location |
| `resighted` | came back to the same object at the same place: no duplicate made |
| `sighting_ended` | object not seen for 1.5 s |
| `deleted` | tombstone from `found()`. It contains only `memory_id`, `label` and `wall_ms`, because the memory's earlier lines are scrubbed |

### Complete sample record (actual output)

From `samples/file_run/memories.json`: `run.py --preset m960d` on the placement clip. The keys were put on a chair at about 4.9 s. The memory was created at 6.0 s and its evidence updated at 6.5 s. It's a file run, so `observed_ms` is null:

```json
{
  "memory_id": "mem_bunch-of-keys_7218baef",
  "label": "bunch of keys",
  "bbox": [695, 755, 907, 919],
  "confidence": 0.148,
  "frame_id": 13,
  "video_ms": 6505,
  "wall_ms": 1792220999254,
  "observed_ms": null,
  "first_seen_video_ms": 5503,
  "last_seen_video_ms": 6505,
  "last_seen_wall_ms": 1792220999254,
  "last_seen_observed_ms": null,
  "sightings": 1,
  "place_status": "first_sighting",
  "scene_verdict": null,
  "scene_inliers": null,
  "compared_with": null,
  "image_path": "images/mem_bunch-of-keys_7218baef.jpg",
  "crop_path": "images/mem_bunch-of-keys_7218baef_crop.jpg",
  "source": "data\\clips\\01_normal_placement.MOV",
  "session_id": "sess_20261017T030934_f48d",
  "source_kind": "file"
}
```

The evidence image is 1920×1080, so the bbox is in those pixels. `samples/live_api_run/memories.json` is the same scene through the API used like a live loop: there `observed_ms` = `wall_ms`, and `time_basis` = `"observed"`.

## 5. How this reaches your backend

Per the contract, your backend doesn't import this package. `lastscene_bridge` turns selected memory events into observations and POSTs them; your existing retrieval stays in charge of answers. Details, an example payload and the decisions we need are in **`LASTSCENE_INTEGRATION.md`**.

Quick test of the ML API without a GPU:
```
python examples/quickstart.py --no-model     # uses a temp copy of samples/file_run; prints every API result
python examples/quickstart.py                # same, with the real detector (downloads the model on first run)
```

## 6. Accuracy and limits (measured on our phone clips; numbers in `eval/`)

- **Presets** (`memory_engine/presets.py`):
  - `efficient` (default, unchanged) works on low-res close-ups. **On wide 1080p frames it misses keys put down on a chair and mistakes laptop keyboards for keys:** 5/5 false memories on the no-keys clip.
  - `m960d` and `x960d` add the "laptop keyboard"/"laptop" distractor labels plus 960 px input: **0 false memories**, and 4/5 and 5/5 useful placement memories respectively. But 960 px hurt on low-res frames.
  - We'll choose after recording the same scenarios with the ESP32 (`eval/manifests/esp32_round1.json`, one command to evaluate). Until then, plan for the preset to be a config value.
- **Reliable:** memory creation on clean surfaces (with the 960 px presets); retrieval with the matching photo, box and time; label-wide deletion; never claiming keys are visible without a live detection.
- **Unreliable:**
  - **cluttered surfaces:** keys tangled with a lanyard or cable score right at the threshold, so a memory needs ~5 s of the keys sitting in view;
  - **far-away keys:** about 30 px, not detected;
  - **scene recognition in plain or repetitive rooms:** identical chairs were recognized in 0–4 of 38 frames while walking back, so far-away guidance falls back to the photo;
  - **duplicate memories** when scene matching can't confirm the same spot.
- **Near-real-time,** not instant: a memory forms about 1–1.5 s after a put-down, at 2 inferences/s.

## 7. Privacy

- The sample images come from Adam's personal phone footage. They show a chair and keys, with no people. **Don't commit `samples/` or anything from `out_dir` to a public repo without Adam's OK.**
- Raw videos are not included.
- The system keeps only evidence frames (no continuous recording), and `found()` deletes them.
