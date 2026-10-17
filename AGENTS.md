# AGENTS.md: context for coding agents (Codex)

Read this before changing anything.
- `backend/` (FastAPI, TigerData), `frontend/` (dashboard) and `device/` (Pi client) are Dylan's.
- `vision/` is Adam's vision pipeline ("LastSeen"), copied in from his separate project on Oct 17. **Its source and tests are now in this checkout:** `vision/memory_engine`, `vision/lastscene_bridge`, `vision/tests`.

## Plan as of now (updated Oct 17, supersedes earlier "no laptop bridge")

- **Demo setup:** the wearable camera streams frames to Adam's RTX 4060 laptop. The laptop runs the vision pipeline and `vision/lastscene_bridge`, and the bridge uploads to Dylan's backend on Vercel (`https://www.lastscene.photo`). This laptop path is also the fallback.
- `python -m lastscene_bridge.run --source <stream-url|clip> --preset m960d` is the demo entry point.
- Pi inference is optional. Benchmark it only on the actual Pi, starting with the unquantized small model; INT8 comes only after that.
- **Caveat from the contract:** YOLO-World probably won't run usefully on a Pi Zero 2 W. Which Pi the team has is unconfirmed.

## What the vision code does

1. YOLO-World (prompt `"bunch of keys"`) detects keys in camera frames, at **2 inferences/s** (soak-tested, GPU near idle on a laptop).
2. `memory_engine` turns detections into **memories**: confirmed across ≥ 2 frames over ≥ 0.5 s, saving the evidence frame + box. ORB scene matching decides whether a new sighting is the same place as an earlier one.
3. `lastscene_bridge` sends one observation per **new** memory to `POST /api/v1/observations` (`backend/app/routers/observations.py`, models in `backend/app/observations.py`).
4. `memory_engine.api.LastSeen.guide()` / `locate()` give live left/right guidance next to the camera. "Found it" deletes local memories and calls `POST /api/v1/items/{label}/found`.

## Layout of `vision/`

```
memory_engine/     ML package. Public API: api.py (LastSeen: process_frame, latest_memory, guide, locate, found)
  presets.py       USE "m960d" (YOLO-World m, 960 px, distractor labels "laptop keyboard"/"laptop").
                   "efficient" (default) fails on 1080p frames and would upload a FALSE "keys" observation
                   on desks: it reads laptop keyboards as keys.
lastscene_bridge/  observation.py (contract JSON + JPEG, bbox px -> normalized [x,y,w,h] via the UPLOADED image size)
                   selector.py (send "created"; end-of-sighting update only if moved >10% or >10 min later)
                   uploader.py, auth.py, capture_time.py (real recording time from video metadata), bridge.py, run.py
tests/             pytest; tests/mock_lastscene.py = strict local mock of the backend + Auth0, from the contract
scripts/           eval_run.py (labeled-clip evaluation), record_stream.py (record ESP32/Pi stream), contact_sheet.py
eval/              latest benchmark (summary + per-preset text reports); eval_manifests/ = labels
docs/              LASTSCENE_INTEGRATION.md (contract audit, decisions), README_HANDOFF.md (ML API + data format),
                   observation_example.json (real payload from a phone clip)
```

**Not in git, on purpose:** the phone videos and evidence images (personal footage; this repo is public), model weights, `.venv`. Some docs mention `samples/` images; those stay on Adam's machine.

## Decisions already made (don't silently reverse them)

- `objects[].event` is always `"seen"`. There's no hand tracking, so placed, moved and picked_up are never proven.
- `scene` is omitted. ORB gives no stable room ids or human labels.
- `captured_at` is real UTC capture time only (device clock, or file metadata for clips). If unknown, don't upload. Never stamp "now" onto old footage.
- `observation_id` is UUIDv5 of (source, capture ms), so retries and re-runs are server-side duplicates (200).
- Confidence is the **raw detector confidence**. Real keys score 0.05–0.35. Don't inflate it.
- Only real detector boxes from the captured frame are uploaded. `guide()`'s projected "they were here" box stays local, and the keys are never "visible now" without a live detection.
- "Found it" deletes **all** local memories of that label, then calls the found endpoint.

## Known integration gaps (deliberately deferred)

1. **Confidence floor.** The backend's `MIN_DETECTION_CONFIDENCE=0.35` returns 422 for our key memories. Deferred: set it to 0.05 on the deployment before any real upload.
2. **Env variable names:** done. The bridge accepts both naming styles (`LASTSCENE_*`/`AUTH0_CLIENT_*` and `API_URL`/`PAIRING_CODE`/`DEVICE_CLIENT_*`).
3. **Base URL** is `https://www.lastscene.photo`. The apex `lastscene.photo` redirects with a 308, and requests drops the bearer token on that redirect.
4. **Token caching:** done. The token is cached to a file, `~/.cache/lastscene/auth0.json`, outside the repo.
5. **Camera stream.** Nothing in the repo streams from the wearable yet. `device/lastscene_device.py` (Dylan) takes one still per button press. The bridge accepts an HTTP MJPEG URL, a webcam index or a file.
6. **Found/reopen.** An old queued sighting uploaded after "found it" clears `found_at` server-side (`upsert_last_seen`). The dashboard doesn't read `found_at` yet.

## How to run the vision tests

```
cd vision
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
python -m pytest -q
```

Expect **`186 passed, 8 skipped`** (Oct 17). The 8 need Adam's local clips and run with `LASTSEEN_E2E=1`. Run tests from inside `vision/`, not the repo root; `backend/tests` is separate. Keep them passing after every change.

Model weights download on first detector use: `yolov8m-worldv2.pt` (~57 MB) and `weights/clip/ViT-B-32.pt` (~354 MB). They're gitignored.

## Don'ts

- Don't commit `.env`, tokens, `data/`, `*.pt`, `weights/`, `upload_queue/` or `memories_out/`.
- Don't change Dylan's API (`backend/app/observations.py`, routers) without asking him.
- Don't add heavier models, or lower the 2 s upload rate limit.
- Don't run two agents on the same files at once.
- Credit AI tools (Claude Code, Codex) in the Devpost submission.

## Known limits

- **Clean surfaces:** works (m960d: useful memory in 4/5 runs on the placement clip).
- **Cluttered surfaces:** unreliable.
- **Far-away keys:** not detected.
- **Room recognition in repetitive rooms:** unreliable.
- **ESP32 / Pi camera:** untested. Evaluate with `scripts/record_stream.py` + `scripts/eval_run.py --manifest eval_manifests/esp32_round1.json`.
