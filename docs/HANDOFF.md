# Handoff: vision pipeline, Oct 17 (Sat, ~9 AM)

Read with [`AGENTS.md`](../AGENTS.md), which has the rules and decisions. Devpost closes **8:00 AM Sunday**.
Branch `ml` = `origin/master` (Dylan's Teach items, React dashboard) merged with the vision work. Nothing is
merged into `master` or deployed.

## Where things stand

**Two ways to run detection, sharing one engine (`vision/memory_engine`) and one uploader (`vision/lastscene_bridge`):**

| Path | Command (from `vision/`) | Model | Status |
|---|---|---|---|
| Laptop demo (fallback) | `python -m lastscene_bridge.run --source <mjpeg-url or clip> --show` | YOLO-World m960d, zero-shot | Works on real clips; scores 0.05–0.35, so the server needs `MIN_DETECTION_CONFIDENCE=0.05` |
| Dylan's spec | `python -m lastscene_pipeline.sync` → `train` → `detect --source …` | YOLO11s fine-tuned on Teach items photos; INT8 ONNX for the Pi | Built and tested offline; never run against the live API |

While either one runs, type `where` / `found` / `quit` in the console, or press `w` / `f` / `q` with `--show`.

### Confirmed (run or read in this session)

- **Deployed backend:** `www.lastscene.photo` is up, with the observation, last-seen, found and training routes. They return 401 without a token.
  - The apex `lastscene.photo` 308-redirects, and requests drops the bearer token on that hop. The code rewrites the apex to `www`.
- **Contract match:** the bridge's payloads pass Dylan's real Pydantic model. That includes `placed`, with and without `scene`.
- **Real local backend run:** the earlier agents ran Dylan's real FastAPI app on a local Postgres 16 + pgvector (Auth0 and Gemini faked). Results: 201, 200 duplicate, 429 then retry, found.
- **Vision tests:** 187 passed, 8 skipped, plus 12 new `tests/test_pipeline.py` tests passing.
  - Real-clip e2e: 7 passed + 1 xfail, before the event-name change.
- **Found it:** stale queued sightings are moved to `superseded/`, and the found call is queued in order and retried. 403 doesn't fetch a new token; a 409 resends under a new stable id.
- **Pilot model** (YOLO11s fine-tuned on 116 frames: 14 keys frames pseudo-labelled by m960d from development clips, plus 102 empty frames; held-out clips never used):

  | | 640 px | 320 px |
  |---|---|---|
  | INT8 size | 10.1 MB | 10.0 MB |
  | Held-out `04_return_and_find` | memory only at floor 0.25 (conf 0.27) | memory at **0.67** (passes 0.35) |
  | Held-out `IMG_7687` (portrait, low-res) | memory at 0.25 floor | missed |
  | No-keys clip `IMG_7688` | 0 false memories | 0 false memories |
  | Cluttered clip | missed | missed |
  | Laptop CPU INT8, 1 / 4 threads | 0.214 / 0.091 s | 0.065 / 0.038 s |
  | Peak memory (`--low-memory`) | 201 MB | 135 MB |

  - The box lands on the keys on held-out footage.
  - INT8 matched 100% of FP32's detections on the calibration photos.
  - `detect` runs with torch and Ultralytics blocked, as on a Pi.
  - The pilot model isn't in git (`lastscene-data/` is ignored). Rebuild it from the Teach items photos instead.

### Not verified (don't claim it)

- Any upload to the deployed backend: there are no device credentials on this machine.
- Pi Zero 2 W speed, memory and picamera2 capture: the Pi is only reachable over SSH from another device. Run [`vision/pi/README.md`](../vision/pi/README.md) step 3 and record the JSON it prints.
- A real wearable stream: nothing in the repo streams yet.
- Accuracy of a model trained on real Teach items photos. The pilot used pseudo-labels from 2 clips.
- The `--show` window, and `sync` against the real training endpoints.

## Decisions made today (Adam)

1. **Dylan's spec wins where it's provable.**
   - Events: `placed` (first confirmed memory), `moved` (ORB says a different place, or the box moved > 10%), `seen` (10-minute resend). Never `picked_up`, because there's no hand tracking.
   - Scene: only from `detect --scene-label` (the operator names where the camera is).
   - `detect` enforces a 0.35 floor.
2. **YOLO11s as the base, not YOLO-World.** It's about 10 MB as INT8 and can't detect an item from its name alone. **Dylan's spec says `--base` must be YOLO-World/YOLOE, so this needs his OK.** On the laptop, `detect` without a trained model still falls back to YOLO-World `set_classes`.
3. **For the Pi, train at `--imgsz 320`** (the 320 px pilot did better than 640 on held-out footage, and runs about 3× faster).

## Next actions

- **Dylan:**
  1. Send the device credentials privately: `DEVICE_CLIENT_ID` and `DEVICE_CLIENT_SECRET` (M2M app authorized for `https://api.lastscene` with `write:memories`), plus the `PAIRING_CODE`.
  2. OK or reject the YOLO11s base.
  3. Review [`docs/for-dylan/NOTE_FOR_DYLAN.md`](for-dylan/NOTE_FOR_DYLAN.md) and the found/reopen fix in [`reopen_fix.patch`](for-dylan/reopen_fix.patch). It was tested on a real Postgres copy, isn't applied, and needs your call.
  4. For the laptop m960d demo only: set `MIN_DETECTION_CONFIDENCE=0.05` on Vercel. Not needed for a trained model.
- **Adam:**
  1. Upload 8–12 photos of the real keys on the Teach items page (several angles and surfaces, keys filling a fair share of the photo).
  2. With credentials, run from `vision/`:
     - `python -m lastscene_pipeline.sync --env-file <file outside the repo>`
     - `python -m lastscene_pipeline.train --imgsz 320`
     - `python -m lastscene_pipeline.detect --source <clip or stream> --dry-run`
     - then `python scripts/live_smoke.py`
  3. On the Pi: `bash pi/install.sh`, copy `models/<version>/` (4 files), run `bench`.
- **Sree / Emilee:** say which camera board actually works. Then either give an HTTP MJPEG URL (`http://<ip>:<port>/stream`, 1280×720 or 1920×1080, 5 fps) with the laptop on the same network, or confirm Pi camera + `--source picam`. Record 30 s with `scripts/record_stream.py`.

## Finish line → code

| Step | Where |
|---|---|
| put keys down → memory | `memory_engine/memory.py` (≥ 2 hits over ≥ 0.5 s) |
| → observation | `lastscene_bridge/bridge.py`, `selector.py`, `observation.py:event_for` |
| → dashboard | `POST /api/v1/observations` → Items/Timeline (device must be claimed into the household first) |
| ask where | `where` command (`lastscene_bridge/live_view.py`), dashboard Search, or `GET /items/keys/last-seen` |
| guidance | `LastSeen.guide()`, drawn by `live_view.py` (projected box labelled "memory", never "visible now") |
| found → clears | `found` command → `bridge.found()` → local delete + queued `POST /items/keys/found`. The dashboard doesn't show `found_at` yet (see the note for Dylan). |

## Scratch state not in git

Pilot dataset/models, the local Postgres used for the patch test, and the dry-run outputs lived in the session
scratchpad (temporary). Recreate the pilot with `train` on real Teach items data rather than reviving it.
