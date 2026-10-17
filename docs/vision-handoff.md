# Vision pipeline handoff: Claude Code → Codex (Sat Oct 17, ~4:30 AM)

Everything built and measured for the vision side so far, where it lives, and what to do next. Read with `AGENTS.md` (rules and decisions) at the repo root.

> **Update (Oct 17, later):** the priority is the live demo on the laptop (see the AGENTS.md "Plan as of now" section). The "Next steps" below are superseded: do Pi benchmarking only on the actual Pi, and no INT8 before unquantized timings exist. Gaps 2 and 3 (env names, token file cache) are done.

## Environment verification update (Codex, Oct 17)

The original `lastseen\.venv` works with Python 3.12.6. The earlier "No Python at"
error came from sandbox access restrictions; no interpreter reinstall was needed.
When Codex cannot access that interpreter, use approved execution outside the
sandbox. Torch 2.11.0+cu128 and the RTX 4060 were verified available.

Installed optional local ONNX tooling from `vision/requirements-onnx.txt`:
ONNX 1.23.2, ONNX Runtime 1.31.0 (CPU), and ONNX Slim 0.1.98. Static INT8 imports
succeed and `pip check` reports no broken requirements. These are Windows build
tools; compatibility and performance on the Pi remain unmeasured.

From `vision/`, run without activating PowerShell scripts:

```powershell
& ..\..\.venv\Scripts\python.exe -m pytest -q
```

After dependency installation: **100 passed, 8 skipped**. This environment check
does not establish successful model export, quantization, or Pi inference.

## Where things are

| What | Location | In git? |
|---|---|---|
| Vision source, tests, eval results, docs | `vision/` in this repo | not yet committed (Adam reviews first) |
| Original project: venv, model weights, phone clips, eval caches | `C:\Users\Administrator\Projects\swamphacks-xii-research\lastseen\` (this repo is cloned *inside* it) | no, keep it that way |
| Weights | `lastseen\yolov8{s,m,x}-worldv2.pt`, `lastseen\weights\clip\ViT-B-32.pt` | no |
| Phone clips (personal) | `lastseen\data\clips\` (01_normal_placement, 03_no_keys = actually clutter WITH keys, second take cluttered, IMG_7688 = actual NO keys, 04_return_and_find, plus older desk_a/desk_b/IMG_7687) | no |
| Working Python env | `lastseen\.venv\Scripts\python.exe` (torch 2.11 cu128, ultralytics 8.4.178) | no |
| Earlier ZIP sent for Dylan | `lastseen\handoff\LastSeen_ML_handoff_2026-10-17.zip` | no |

Tests: run `python -m pytest -q` from `vision/`. Expect `100 passed, 8 skipped`. The 8 run with `LASTSEEN_E2E=1` and need the clips at `vision/data/clips`, or run them in `lastseen\`.

## What's done and verified (laptop, RTX 4060, phone footage)

- **Detection:** YOLO-World with the prompt `"bunch of keys"` (far better than `"keys"`).
  - Presets live in `memory_engine/presets.py`.
  - **`m960d`** (m model, 960 px, distractor labels "laptop keyboard"/"laptop", FP16) is the best measured option for 1080p frames.
  - `efficient` (m, 480 px) is the code default but **fails on 1080p**: no useful placement memory, and it reads laptop keyboards as keys (false memories on the no-keys clip in 5 of 5 sampling-timing variations).
- **Memory engine:**
  - confirms across ≥ 2 detections over ≥ 0.5 s at 2 inferences/s;
  - saves the evidence frame + box, with the latest-sighting photo;
  - ORB scene matching avoids duplicates when you return to the same textured spot; "unknown" is never treated as a new place;
  - label-wide deletion on "found it".
- **Results** (`vision/eval/summary_phone_round2.md`), m960d at 0.05:
  - useful key memory before walking away: 4/5 runs on the chair clip;
  - **0 false memories** on the no-keys clip;
  - memory forms about 1–1.5 s after the keys are put down;
  - cluttered table 0/5 at 0.05 (works at 0.03 but makes duplicates);
  - far-away keys never detected;
  - room recognition 0–4 of 38 frames on identical chairs.
- **Thermals:** 2 inferences/s for a 10-minute soak, about 6 W average, 55 °C peak, no errors.
- **Contract bridge** (`lastscene_bridge/`):
  - observation JSON + full JPEG; box normalized via the uploaded image size;
  - UUIDv5 ids (stable across retries and re-runs);
  - `captured_at` from the device clock or the video's metadata, never invented; `event` is always `"seen"`;
  - Auth0 + `X-Pairing-Code`, 2 s rate limit, retries, an offline queue in capture order.
- **End to end against `tests/mock_lastscene.py`** (strict mock, written from the contract): a real clip went through the real detector to an observation and a `201`. The mock's last-seen for "keys" had `captured_at` = recording time + 6.0 s, and the box landed on the keys.
- **Checked against Dylan's actual backend code** (`backend/app/observations.py`): the field format matches. Not tested against his deployed server.

## Known gaps (deferred on purpose)

1. **The backend rejects our key memories:** `MIN_DETECTION_CONFIDENCE=0.35` returns a 422, and real keys score 0.05–0.35. One env line (0.05) before any real upload.
2. **Env names:** the bridge uses `LASTSCENE_*` / `AUTH0_CLIENT_*`; Dylan's scripts use `API_URL` / `PAIRING_CODE` / `DEVICE_CLIENT_*`. Accept both.
3. **Base URL:** `https://www.lastscene.photo`. Auth0 M2M tokens should be cached to a file (free-plan cap). The token needs the `write:memories` scope.

## Review of `docs/item-enrollment-pi-spec.md` (Codex, earlier tonight)

The spec is careful and honest. But for the ~28 h left (Devpost closes **8:00 AM Sunday**), it's scoped too big, and it rests on an unmeasured assumption:
- **"Pi Zero 2 W" isn't confirmed.** Adam answered "yes, idk". Ask Sree or Emilee which Pi they actually have.
- **The m model at 960 px is unlikely to run usefully on a Zero 2 W** (512 MB RAM, 1 GHz A53). This is an estimate: tens of seconds per frame vs. the 2/s the memory logic is tuned for, and possibly out of memory.
- **Shrinking doesn't save it:** our benchmark showed the small model at 320 px detects **0%** of placed keys.
- **INT8 may push key confidences** (already 0.05–0.35) under the threshold.
- **Enrollment, identity matching and package sync are weeks of work.** Keep them as "future work" for the Devpost.

## Next steps, in this order

1. **Measure before converting (about 20 min).**
   - On the laptop: `YOLO("yolov8s-worldv2.pt")` → `set_classes(["bunch of keys","laptop keyboard","laptop"])` → `export(format="onnx", imgsz=640)` (then 320).
   - Copy it to the **actual Pi**. Time 10 frames with onnxruntime, and record peak memory.
   - **Kill rule:** more than ~2 s per frame, or out of memory → **the Pi only captures and uploads frames; detection runs on the laptop or a worker.** The bridge code works unchanged wherever the detector runs.
2. **Only if it passes:** INT8 static quantization, calibrated on frames from different clips than the ones you test on. Then re-run `scripts/eval_run.py --manifest eval_manifests/phone_round2.json`. If INT8 loses the keys, use FP32.
3. **Get one real automatic key memory onto Dylan's dashboard.** Set his deployment's `MIN_DETECTION_CONFIDENCE=0.05`, real Auth0 device credentials and a pairing code, then run the bridge (`m960d`) on a clip or live camera. This is the demo's core.
4. **Demo keys only.** Other items are experimental until each has its own prompt test plus a no-false-alarm clip. Simple "add item" = add a prompt to a config list, labeled untested.
5. **Record the demo scenarios on the real Pi or ESP32 camera** (`scripts/record_stream.py`), fill in `eval_manifests/esp32_round1.json`, and run `scripts/eval_run.py` to pick the final preset.

## Don'ts (summary; full list in AGENTS.md)

- Don't commit clips, evidence images (public repo), `*.pt`, `.env` or tokens.
- Don't inflate confidence values to pass the 0.35 floor.
- Don't change Dylan's API without asking him.
- Don't claim untested items or Pi performance.
