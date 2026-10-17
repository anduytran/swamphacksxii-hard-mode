# LastSeen: memory engine (Adam's component)

Turns a video or live camera into **object memories**: "keys were seen here, in this image, at this time."

Built at SwampHacks XII with help from Claude Code (Anthropic). Uses Ultralytics YOLO-World (AGPL-3.0) and OpenCV.

## Setup

```
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
pytest -q
```

## Run

```
python -m memory_engine.run --source data\clips\keys_desk.mp4 --prompts "keys,wallet" --show
python -m memory_engine.run --source 0                                  # webcam
python -m memory_engine.run --source http://<esp-ip>:81/stream          # ESP32 camera (untested: no stream yet)
```

By default the detector runs **2 times per second** (`--inference-fps`), whatever the camera's frame rate. Video files play in real time, so the loop waits between inferences like a live camera would; `--fast` disables that for offline evaluation.
- Presets: `--preset efficient` (default; m, 480 px, FP16), `light` (s), `xl` (x).
- `--gpu-log gpu.csv` records GPU utilization, power and temperature.
- At ≤ 2 inferences/s the memory thresholds adapt automatically: 2 hits instead of 3, and the gap tolerance becomes ≥ 3 inference intervals.

## Output (what the backend reads)

```
<out>/memories.json      current state of every memory, rewritten atomically on each change
<out>/events.jsonl       append-only log, one event per line
<out>/images/<memory_id>.jpg        full scene, no drawings (draw the bbox yourself)
<out>/images/<memory_id>_crop.jpg   close-up of the object
```

One memory in `memories.json`:
```json
{
  "memory_id": "mem_keys_3f9a2c1b",
  "label": "keys",
  "bbox": [402, 288, 450, 310],
  "confidence": 0.47,
  "frame_id": 1901,
  "video_ms": 63366,
  "wall_ms": 1792206736824,
  "first_seen_video_ms": 62833,
  "last_seen_video_ms": 71200,
  "sightings": 2,
  "image_path": "images/mem_keys_3f9a2c1b.jpg",
  "crop_path": "images/mem_keys_3f9a2c1b_crop.jpg",
  "source": "data\\clips\\keys_desk.mp4"
}
```

- `bbox` is x1, y1, x2, y2 in pixels of the saved image.
- `video_ms` is when the saved image was taken: the position in the video for files, or time since the stream opened for live sources. **It's only comparable within one session.**
- `wall_ms` is when that frame was *processed*, as Unix ms.

Fields added later. They're additive, and the ones above are unchanged:
- `observed_ms` / `last_seen_observed_ms`: real-world time the scene was *seen*. For live sources this equals the wall clock. For files it's `--recorded-at` + `video_ms`, or `null` if no recording time was given. Use these to compare sightings across sessions.
- `last_seen_wall_ms`
- `session_id`: one per run, e.g. `sess_20261016T234957_78f8`.
- `source_kind`: `file` or `live`.

**The evidence image always shows the latest location.** When a sighting ends, the saved image switches to the last frame of that sighting with confidence at least 60% of the sighting's best. It does this even if an earlier frame was clearer. `video_ms`/`bbox` always describe the saved image; `last_seen_*` says when the object was last detected, which may be a weaker frame after that.

## Compare two real frames (scene matcher check)

```
python scripts\compare_frames.py --a data\clips\look_away.mp4 --a-time 1.0 --b-time 9.5 --detect keys
python scripts\compare_frames.py --a data\clips\walk_around.mp4 --a-time 1 --sweep 0.5 --detect keys
```

It prints the verdict, reason, inlier counts and time, and saves side-by-side images to `data\compare\`: matched features as lines, the box in A in red, the projected box in B in green.

Event types in `events.jsonl`:

| event | meaning |
|---|---|
| `created` | new object/place, image saved |
| `image_updated` | a clearer frame of the same memory replaced the image |
| `resighted` | came back to the same object in the same place (no duplicate made) |
| `sighting_ended` | the object hasn't been seen for `max_gap_ms` |

From Python, the backend can skip files entirely: `MemoryMaker.process(frame_id, video_ms, wall_ms, detections)` returns the same event dicts.

## How a memory gets made (memory_engine/memory.py)

1. Drop detections below `min_conf` (0.25).
2. Match each detection to a **track**: the same label with an overlapping box (IoU ≥ 0.3) as in recent frames.
3. A track becomes a **memory** after 3+ detections spread over ≥ 500 ms. One-frame flukes never qualify.
4. **No duplicates:** if a memory with the same label already has an overlapping box (IoU ≥ 0.5), reattach to it (`resighted`) instead of creating a new one.
5. A track ends after 1500 ms unseen.

All thresholds are in milliseconds, so a 5 fps ESP32 stream and a 30 fps phone clip behave the same.

## Scene matcher (memory_engine/scene.py)

This answers "is this the same place as that old memory photo?" using ORB features + a RANSAC homography. No training, OpenCV only.

- **Verdicts:**
  - `same_scene`: enough well-spread matching features.
  - `different_scene`: both images are well textured, but almost nothing matches. That's evidence, not proof.
  - `unknown`: anything else, such as too little texture or too few/clustered matches.
- **`project_box`** maps an old box into the new frame. It works only when `projection_trusted` is true, which requires passing sanity checks: no mirroring, zoom within 3×, limited tilt, no folding.

Measured on this laptop (`python scripts/bench_scene.py`, RTX 4060 laptop; this all runs on the CPU):
- features() takes 8.7 ms for a 640×480 frame;
- one match() takes 6.5 ms.

On random flat warps of one photo:
- up to 30% corner shift: 40/40 `same_scene`, with trusted boxes within 3.2 px;
- at 40%: 33 same, 2 unknown, **5 wrongly `different_scene`**, and the worst trusted box was off by 9.2 px.

These are synthetic, flat warps, so treat them as a best case.

## Real-clip results (Oct 17, three phone clips in data/clips)

The clips (all H.264, portrait; OpenCV reads every frame):
- `desk_a.mov`: keys on a chair seat in a room of identical chairs. Camera turns away, then views it again from across the room. 5.3 s, 360×640.
- `desk_b.mov`: keys on a white table with a laptop. Steady camera. 6.7 s, 360×640.
- `IMG_7687.MP4`: keys on patterned carpet, camera moving around. 7.8 s, 720×1280.

**Detection (YOLO-World s, sampled every 6th frame):**
- `"keys"` was found in 0/53, 0/66 and 5/39 frames (confidence ≤ 0.14).
- `"bunch of keys"` was best: 11–17/53 on the chair, 0/66 on the table, 17–22/39 on the carpet, median confidence about 0.1.
- 1280-px input found nothing.

**Real keys are not reliably detected by this model.** This is the main open risk.

**Larger model (`yolov8x-worldv2`, same family), prompt `"bunch of keys"`:**
- Found the keys in 17/53, 10/66 and 29/39 sampled frames.
- Up close (desk_a 0–1.6 s): 32/32 frames, with **0 false alarms in 40 frames where the keys were out of view**.
- It still misses keys that are far away (desk_a's return from across the room) and keys tangled in a lanyard (desk_b, confidence 0.05–0.08).
- Speed: about 31–35 fps vs. about 86–90 fps for the small model.

With `--min-conf 0.05`, every clip yields exactly one correct key memory, after adding the oversized-box filter (`drop_oversized`, >50% of the frame). Without that filter, a whole-frame box at 0.08 became a false memory. Phone, wallet, sunglasses and remote produced 0 false alarms on these clips. Their *detection* is untested: none of those objects appear in the clips.

**Scene matching** (surroundings only: the object box is excluded from matching):

| Test | Result |
|---|---|
| Different locations (desk_a↔desk_b, desk_b↔carpet; 30 comparisons) | 30/30 `different_scene`, 0 false "same" |
| Carpet, camera moving around the keys (31 frames) | 26 same, 5 unknown. Projected box vs. the detector's own box in the 8 frames where it fired: IoU median 0.875, min 0.79 |
| Table, steady camera (14 frames) | 14/14 same |
| Chair: close-up, look away, return from across the room (22 frames) | 3 same (close-ups only), 14 unknown, 5 `different_scene`. **The return from far away was not recognized.** Scale changed a lot, and the plain chair seat has little texture of its own. |

Earlier, with object features included, the matches on the carpet clip sat mostly on the keys themselves. That would make "same scene" mean "same object," so surroundings-only is now what the compare tool does by default.

Speed: about 7 ms per comparison for 360×640 frames, about 40 ms for 720×1280 (this laptop, on battery).

## Detector benchmark (`scripts/bench_detector.py`, prompt "bunch of keys")

**How it was scored:**
- The keys' visibility was labeled by eye in time windows: desk_a close (0–1.7 s), out of view (1.9–3.5 s), far (3.8–5.3 s); desk_b (table) and carpet visible throughout.
- Every 3rd frame was scored: 34 near, 30 far, 131 table, 78 carpet, 31 absent.
- Recall = labeled frames with a box at confidence ≥ 0.05 / ≥ 0.15.

| config | near | far | table | carpet | false alarms (31) | ms/frame (peak) | peak W | key memories at 2 fps (a/b/carpet) |
|---|---|---|---|---|---|---|---|---|
| s 640 | 67/11 | 0/0 | 0/0 | 60/16 | 0 | 10.6 | 36 | 0/0/1 |
| s 640 FP16 | 67/11 | 0/0 | 0/0 | 57/15 | 0 | 10.5 | 29 | 0/0/1 |
| s 480 FP16 | 91/32 | 0/0 | 0/0 | 50/11 | 0 | 11.6 | 25 | 1/0/2 |
| s 320 FP16 | 79/29 | 0/0 | 0/0 | 47/5 | 0 | 10.5 | 20 | 0/0/1 |
| **m 640 FP16** | **100/73** | 0/0 | 0/0 | **91/41** | 0 | 13.1 | 40 | 1/0/1 |
| m 480 FP16 | 97/50 | 0/0 | 26/0 | 87/26 | 0 | 13.6 | 31 | 1/1/1 |
| x 640 | 100/97 | 0/0 | 15/0 | 76/19 | 0 | 30.4 | 58 | 1/1/1 |
| x 640 FP16 | 100/97 | 0/0 | 11/0 | 75/19 | 0 | 18.4 | 54 | 1/1/1 |
| x 480 FP16 | 100/100 | 0/0 | 7/0 | 92/44 | 0 | 18.7 | 46 | 1/0/1 |
| cascade s→x | 85/85 | 0/0 | 0/0 | 73/23 | 0 | 14.8 (x on 29%) | 41 | 0/0/1 |
| cascade s→x + x every 4th | 91/88 | 0/0 | 2/0 | 75/23 | 0 | 13.4 (x on 46%) | 47 | 0/0/1 |

Notes:
- "Peak" = frames run back-to-back. Those milliseconds and watts are not what an always-on wearable sees; see the sustained test below.
- FP16 (`quantize=16`) cut x from 25 to 15.6 ms per frame with identical detections (59/78 carpet frames).
- The small model's time is dominated by pre/post-processing on the CPU, so FP16 doesn't speed it up.
- **The cascade didn't pay off.** The small model often sees *nothing* (not even a weak candidate) where the big one would succeed, so there's nothing to escalate. It ends up below the medium model on its own.
- **Far keys: 0% for every config.** In desk_a's far frames the keys are about 30×25 px, since that clip is only 360×640, and best confidence was ≤ 0.017. The size filter removed **0** boxes in far frames, so it is not the cause. (It does remove whole-frame boxes in close-ups, where the correct box is still kept.) Splitting the frame into 2×2 overlapping tiles found them in 7/30 frames with x (5/30 with m), with correct boxes. That's 4× the compute, so it only makes sense for an on-demand search.
- **Table (desk_b):** keys tangled with a lanyard, a flashlight and a cable are missed by almost everything (best 26%).

**Sustained load at 2 inferences/s** (`scripts/bench_sustained.py`, 60 s each, paced with sleeps, nvidia-smi sampled once per second):

| | ms/inference | GPU busy | mean W | max W | temp start → max |
|---|---|---|---|---|---|
| idle | – | – | 6.5 | 14.3 | 52 °C |
| s 640 FP16 | 17.6 | 3.5% | 11.6 | 17.3 | 51 → 54 °C |
| m 640 FP16 | 58.7 | 11.7% | 6.5 | 13.6 | 53 → 53 °C |
| x 640 FP16 | 72.8 | 14.6% | 7.6 | 20.4 | 48 → 50 °C |

- At 2 inferences/s every config stays close to idle. The heat came from running x back-to-back, at roughly 30 inferences/s and 54–58 W.
- Paced inferences are *slower* than back-to-back ones (e.g. m: 59 vs 13 ms) because the GPU drops its clocks between calls. That's fine at 2/s.
- Sampling once per second can't resolve 60 ms bursts, so the per-config power differences here are within noise.

**Chosen default: `--preset efficient` = m, 480 px, FP16, 2 inferences/s, min_conf 0.05.**
- Recall: near 97%, carpet 87%, and the only config besides x to find the tangled table keys at all (26%).
- 0 false alarms.
- It's the only small or medium config that created a key memory on all 3 clips. All three were checked visually and are correct.
- It's not the most accurate: m 640 has better ≥0.15 confidence, and x has the best close-up recall. But it's the best trade-off we measured. `--preset xl` stays available.

The desk_b memory appeared only at 5.5 s with confidence 0.05, right at the threshold, so cluttered scenes remain fragile.

**10-minute soak** of the default (m 480 FP16, 2 inferences/s):
- 1,201 inferences in 600 s, no errors.
- Temperature averaged 48–51 °C in every quarter and peaked at 55 °C, with no upward trend.
- Mean power 5.7 W, with spikes up to 43 W.
- No thermal-slowdown flag active. That flag was read after the run, not during it.

## Scene matching in memory creation (`--scene-matching on`, the default)

When a new sighting is confirmed, the engine compares the frame's **surroundings** (the object box is masked out) with each remembered evidence image of the same label, up to the 5 most recent. If it's the same scene and the remembered box, mapped into the new frame, lands on the new detection, that's the same place: `resighted`. Otherwise it creates a new memory with an honest `place_status`:
- `likely_new_place`: every comparison said different scene.
- `moved_in_scene`: same scene, but the object is somewhere else in it.
- `unverified`: couldn't tell.
- `first_sighting`: nothing to compare against.
- `another_instance`: two objects with the same label in view at once.

Failed matches are never treated as proof of a new place, and no memory is ever marked "not there".

**A/B on real frames** (`scripts/ab_scene.py`, 4 sessions stitched from the phone clips at 2 fps; identical detections in both modes):

| Session | scene OFF (image position) | scene ON |
|---|---|---|
| S1 same carpet spot, looked away, came back from another position | **2 memories (duplicate)** | **1 memory, resighted** |
| S2 keys moved chair → table | 2, latest = table ✓ | 2, latest = table ✓, `likely_new_place` |
| S3 table shifted so the keys sit at the chair's pixels (constructed) | 2 ✓ | 2 ✓, `likely_new_place` |
| S4 chair close-up → away → far return | 1 ✓ | 1 ✓ |

Fully correct: OFF 3/4, ON 4/4.
- The real-data difference is S1.
- In S3 the old rule happened to pass, because the table box (keys + lanyard) is much bigger than the chair box. The "same pixels, different place" trap is only demonstrated in the controlled-box unit tests (`tests/test_memory_scene.py`).
- In the first S3 attempt the detector found **0** keys in the shifted table frames (black border + clutter). So that session now runs detection on the unshifted frames and moves the boxes.

## Cluttered table (known failure mode)

- **The numbers:** on desk_b, keys tangled with a lanyard, cable and flashlight score 0.02–0.06 at 2 fps. At the default min_conf 0.05, the first memory forms only after **5.3–5.5 s** of the keys sitting in view. At 0.03 it forms after 1.0–1.8 s; at ≥ 0.08, never.
- **What it means:** a quick put-down and walk-away on a cluttered surface would be missed.
- **Why 0.05 stays the default:** 0.03 produced 0 false alarms in 47 out-of-view frames, but those frames were walls, chairs and carpet. The real risk at lower thresholds is desk clutter *without* keys, which we haven't recorded.
- **Tests:** `tests/test_e2e_real.py::test_cluttered_table_memory_within_3_seconds` is an expected failure (xfail) documenting this.

## Real-world evaluation, round 2 (five 1080p phone clips, Oct 17)

Clip names were corrected by Adam; the files are untouched and the scripts map them:

| File | What it really is |
|---|---|
| `01_normal_placement.MOV` | **01** carry keys, put them on a chair seat (≈4.9 s), look away (7.6–16.2 s), come back (16.5 s) |
| `03_no_keys.MOV` | **02 cluttered placement**: it *does* contain keys (table, next to a can), visible 0–2.6 s |
| `second take cluttered.MOV` | **02b** cluttered table, 2nd take, keys visible 0–6.6 s |
| `IMG_7688.MOV` | **03 no keys**: the false-positive test (31.6 s cluttered desk, laptops, cans, phones) |
| `04_return_and_find.MOV` | **04** walk back from across the room (keys tiny until ≈4.4 s), pick them up at 8.3–9.2 s |

Reproduce with one command (labels live in `eval_manifests/phone_round2.json`):

```
python scripts\eval_run.py --manifest eval_manifests\phone_round2.json --presets efficient,m480d,m640d,m960d,x960d
```

Results (`data/eval/phone_round2/summary.md`). Detection runs on full-resolution frames. The production memory engine runs at 2 inferences/s with scene matching on, and every number is out of 5 sampling phases:

| preset | ms/inf | 01 placement: useful memory @0.05 (@0.03) | 02 clutter | 02b clutter | **03 no keys: false memories** | 04 return: memory formed | retrieval | close-up live detection (04) |
|---|---|---|---|---|---|---|---|---|
| efficient (default, m480) | 20 | **0/5** (1/5) | 0/5 (0/5) | 1/5 (1/5) | **5/5 (5/5)** | 0/5 (2/5) | **wrong: keys in hand** | 1/38 |
| m480d | 15 | 2/5 (2/5) | 3/5 (5/5) | 0/5 (5/5) | 0/5 (0/5) | 2/5 (5/5) | OK | 6/38 |
| m640d | 15 | 3/5 (5/5) | 2/5 (5/5) | 1/5 (5/5) | 0/5 (0/5) | 5/5 (5/5) | OK | 16/38 |
| m960d | 16 | 4/5 (4/5) | 0/5 (3/5) | 0/5 (5/5) | 0/5 (0/5) | 4/5 (5/5) | OK | 15/38 |
| x960d | 29 | **5/5** (5/5) | 0/5 (2/5) | 0/5 (3/5) | 0/5 (1/5) | 5/5 (5/5) | OK | 34/38 |

In every preset:
- frames claiming the keys were visible without a live detection: **0**;
- label-wide "found it" deletion: **OK**.

Scene recognition while walking up in 04 stayed at 0–4 of 38 frames for all presets.

At 0.03, the clutter clips form memories more often, but with 2–4 duplicate memories per clip.

Key findings:
- **m480 fails on wide 1080p frames.** Keys on a seat end up about 35 px at the model's 480 px input.
- **m480 also reads laptop keyboards as "bunch of keys",** creating a false memory in every phase of the no-keys clip, at both thresholds. Adding "laptop keyboard"/"laptop" as distractor classes, with class-agnostic NMS, removed those false memories. The two remaining one-frame false detections (m960d / x960d) are cable plugs.
- **960 px hurts on low-res frames.** On the earlier 360×640 clips, m960d dropped from 26% to 0% on the table and from 87% to 29% on the carpet. The right input size depends on the camera, so it must be chosen on real ESP32 frames.
- **Scene matching** never recognized the remembered seat while walking up in 04: identical gray chairs and repetitive carpet, with the keys masked out. It did recognize it after the pickup, with 103 inliers, and projected the box onto the empty seat. **No frame claimed the keys were visible without a live detection** (0 in every config).
- **2 inferences/s** isn't the main limit. At 5–10/s, first-memory times were similar. But at 2/s the outcome depends on the sampling phase whenever confidences hover near the threshold.

Defaults are unchanged pending ESP32 footage (`efficient` = m480, distractors off). The candidates are available as `--preset m480d / m640d / m960d / x960d` (`memory_engine/presets.py`).

## Evaluating the ESP32 camera (as soon as Sree's stream works)

1. **Record** the five scenarios listed in `eval_manifests/esp32_round1.json`, with the camera worn as in the demo:
   ```
   python scripts\record_stream.py --source http://<esp-ip>:81/stream --name 01_placement --seconds 30 --show
   ```
   It writes `data/clips/esp32/01_placement.mp4` plus `01_placement.json`, which reports resolution, measured fps, frame-interval jitter, stalls over 0.5 s and reconnects. That file doubles as the stream-quality report for Sree.
2. **Label:** `python scripts\contact_sheet.py data\clips\esp32\01_placement.mp4` (add `--from 4 --to 6 --every 0.2` to zoom in on a moment). Fill in the `windows` and the `placed_at` / `walk_away` / `first_visible` times in the manifest. Unlabeled clips still run; they just aren't scored.
3. **Evaluate:**
   ```
   python scripts\eval_run.py --manifest eval_manifests\esp32_round1.json --presets efficient,m480d,m640d,m960d
   ```
   Then read `data/eval/esp32_round1/summary.md`.
4. **Pick the default** from that table:
   - first requirement: zero false memories on `03_no_keys`;
   - then: the most useful placement memories;
   - then: the fewest duplicates.

Dry-run on our phone clips (recorded via `record_stream.py` from files, unlabeled manifest): recording, contact sheet, detection cache, analysis and summary all ran.

## For Dylan: the ML interface (`memory_engine/api.py`)

Plain Python, called in-process. Dylan's endpoints can wrap each method one-to-one.

```python
from memory_engine.api import LastSeen, resolve_label
ls = LastSeen(out_dir="data/memories_out")              # efficient preset, scene matching on
ls.process_frame(frame_bgr, video_ms)                   # camera loop, 2x per second
mem = ls.latest_memory(resolve_label("my keys"))        # "where are my keys?"
g = ls.guide(frame_bgr, mem["memory_id"])               # while the user looks around
hit = ls.locate(frame_bgr, mem["label"], search=True)   # thorough on-demand search (XL + tiles)
ls.found(label=mem["label"])                            # "found it": clears EVERY memory of that object
```

On "found it", call `found(label=...)`, not `found(memory_id)`. Scene matching can't always tell "same spot from another angle" (it failed on identical chairs). So the same keys can have an older `unverified` memory, which would otherwise resurface as the next answer. This happened with x960d on clip 01.

| Method | Returns |
|---|---|
| `latest_memory(label)` | The stored record plus `image_file`, `crop_file`, `caveats` (list of sentences), `alternatives` (older memory ids) and `time_basis` (`observed` or `processing`). The most recent sighting wins over the clearest one. |
| `guide(frame, memory_id)` | `status`: `object_at_remembered_spot`, `remembered_spot_in_view`, `remembered_spot_out_of_frame`, `object_seen_elsewhere`, `object_detected` or `remembered_place_not_recognized`. Also `target_box` (draw it), `direction` (`offset_x/offset_y` from -1 to 1, `horizontal` left/center/right, `vertical` up/middle/down, `in_frame`), `remembered_box_in_view`, `scene_verdict` and `detection`. |
| `locate(frame, label, search=False)` | `visible`, `bbox`, `confidence`, `direction`. `search=True` loads the XL model on first use. |
| `found(memory_id or label=...)` | Deletes both images, the record and its history in `events.jsonl` (leaving a tombstone), plus cached scene features. Pauses new memories of that label for 5 s. |

`resolve_label` maps "keys", "my keys", "car keys" and similar to the detector prompt `"bunch of keys"`. File-based access still works: `memories.json` keeps every old field and adds `place_status`, `scene_verdict`, `scene_inliers` and `compared_with`.

## Known limits (measured or reasoned, not hidden)

- The scene matcher hasn't been tested on real 3D scenes. Parallax, clutter, reflective surfaces and ESP32 blur will all reduce matches. Very large viewpoint changes can produce `different_scene` for the same place (seen at 40% warps).

- **"Same place" means the same position in the image.** That works for a steady camera. For a worn camera that moves:
  - looking back at the same keys from a different angle can create a duplicate;
  - a very large object shifted a little can still count as the same place. Seen in testing: a bus shifted 300 px was still IoU ≥ 0.5.

  Fix later: compare the scenes themselves, not just the box position.
- There are no hands yet, so a memory says "seen here," not "placed here."
- Test results so far come from a staged video (a photo shown, hidden, re-shown, shifted) to check the logic. They're not an accuracy measurement on real objects.
