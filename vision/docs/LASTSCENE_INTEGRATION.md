# LastSeen → LastScene integration (our side of the Oct 17 contract)

This covers the vision-pipeline side of "LastScene × vision pipeline: integration contract":
- turning confirmed key memories into `POST /api/v1/observations` uploads (full JPEG + observation JSON);
- the client for the last-seen, found and delete endpoints.

**Tested against a local mock written from the contract, not against Dylan's server.** The contract lists those endpoints under the backend's next steps, so nothing was sent to `lastscene.photo`.

## 1. Field-by-field audit: our memory record → contract observation

| Contract field | Required | Our source | Status / conversion |
|---|---|---|---|
| `image` part (full frame, JPEG/PNG ≤ 2 MB) | yes | evidence image `images/<memory_id>.jpg` (full, uncropped camera frame) | **compatible.** 1920×1080 frames are about 370 KB. Bytes are snapshotted when queued, because our file is overwritten when evidence updates. Re-encoded/downscaled only if > 2 MB, or after a 413 |
| `observation_id` (UUID) | yes | none | **generated:** UUIDv5 of (source, observed_ms of the evidence frame). Stable across retries **and** re-runs of the same clip; verified, since the dry run and the real upload produced the same id |
| `captured_at` (ISO 8601 UTC) | yes | `observed_ms` | **converted:** ms → `2026-10-17T05:36:00.003Z`. Live camera: host clock. Recorded clip: the file's own metadata time (QuickTime `mvhd`) + position in the clip. **No real time known → memory stays local, nothing is uploaded** (`skipped_no_capture_time`) |
| `source` | yes | none | constant `"auto"` |
| `pipeline` {name, version, detector, matcher} | yes | preset | `{"name":"lastseen-ml","version":"0.1.0","detector":"yolo-world yolov8m-worldv2 960px+distractors (preset m960d)","matcher":"orb"}` |
| `image_size` {width, height} | yes | none | **measured from the JPEG actually uploaded** (and updated if it's re-encoded) |
| `scene` {id, label, match_score} | no | `scene_verdict`, `scene_inliers`, `compared_with` | **omitted.** We have no stable scene id and no human label (decision 3 below) |
| `objects[].label` (lowercase singular) | yes | `label` = `"bunch of keys"` (detector prompt) | **mapped** → `"keys"` |
| `objects[].confidence` (0–1) | yes | `confidence` (detector) | **compatible format, threshold conflict** (decision 1). Typical real values are 0.05–0.35; ours: 0.054 |
| `objects[].bbox` [x, y, w, h] normalized | yes | `bbox` [x1, y1, x2, y2] pixels | **converted:** the box is scaled to the uploaded image, then divided by *its* width/height, and clipped to 0–1 |
| `objects[].track_id` | no | `memory_id` | `mem_bunch-of-keys_6efe2634`: stable for that object at that place |
| `objects[].event` placed/seen/moved/picked_up | yes | `event` + `place_status` | **always `"seen"`.** We confirm the object is there, but can't prove a placement, move or pickup (no hand tracking). `seen` still updates the backend's last-seen |
| `audio` | no | none | not sent |

Fields we have and the contract doesn't take: `place_status`, `scene_verdict`/`scene_inliers`/`compared_with`, `sightings`, `first_seen_*`/`last_seen_*`, `crop_path`. They stay local. They aren't sent because the contract lists no extension fields, and a strict validator would reject unknown fields. Our mock does.

## 2. What gets sent, and when ("send events, not frames")

| Our memory event | Sent? |
|---|---|
| `created` (first confirmed sighting of this object here) | **yes** |
| `image_updated` at the end of a sighting | only if the evidence **moved** more than 10% of the frame since the last sent observation for that memory, or **more than 10 minutes** passed (the contract's "seen again after more than 10 minutes") |
| `image_updated` mid-sighting (just a clearer frame) | no |
| `resighted` | no. It carries the *old* evidence frame; the fresh frame at sighting end is judged by the rule above |
| `sighting_ended` | no |

- Objects from the same frame are batched into one observation.
- Processed frames are never uploaded, only these events.
- **Effect:** the backend's last-seen time is the *confirmation* frame. On clip 01, the observation is from 6.0 s at confidence 0.054. Locally, a clearer frame of the same spot (6.5 s, 0.148) later replaced the evidence and was **not** re-sent. This is tunable if Dylan prefers the latest frame.
- **Client-side rate limit:** one observation POST per 2.05 s. A `429` honors `Retry-After`, with at most 3 waits before backing off.

## 3. Example payload (actual output)

Clip `01_normal_placement.MOV`, preset `m960d`, recording time from the file metadata (05:35:54Z). The keys were put on a chair at about 4.9 s and confirmed at 6.0 s:

```json
{
  "observation_id": "8c955a6f-55b0-5348-ad86-777c6b23dcf4",
  "captured_at": "2026-10-17T05:36:00.003Z",
  "source": "auto",
  "pipeline": {"name": "lastseen-ml", "version": "0.1.0",
               "detector": "yolo-world yolov8m-worldv2 960px+distractors (preset m960d)", "matcher": "orb"},
  "image_size": {"width": 1920, "height": 1080},
  "objects": [
    {"label": "keys", "confidence": 0.054, "bbox": [0.3797, 0.637, 0.0885, 0.1565],
     "track_id": "mem_bunch-of-keys_6efe2634", "event": "seen"}
  ]
}
```

Files in `samples/lastscene_observation/`:
- the JSON and the exact JPEG that was uploaded;
- `*_preview.jpg`, with the box drawn **from the normalized values**: it lands on the keys.

## 4. Uploader behavior (`lastscene_bridge/uploader.py`)

- **Auth:** an Auth0 client-credentials token, cached until 60 s before expiry and refreshed once on a 401/403. The `X-Pairing-Code` header is sent on every request.
- **Offline queue:** observations wait on disk in `queue/pending/` (they survive restarts) and are sent in `captured_at` order.
  - Network errors and 5xx: back off exponentially (capped at 60 s), stopping at the head of the queue to preserve order.
  - `201` → `sent/`; `200` (duplicate id) → `sent/`.
  - `422` → `failed/`, with the server's message.
  - `413` → shrink and retry, up to 3 times.
- **Other endpoints:** `found(label, found_at, observation_id)`, `last_seen(label=None)`, `delete_observation(id)`.
- **Credentials:** environment variables only (`.env.example` lists the names). No defaults: with `LASTSCENE_API_BASE` unset, nothing is sent.

Run against a backend:
```
python -m lastscene_bridge.run --source http://<esp-ip>:81/stream --preset m960d          # live
python -m lastscene_bridge.run --source clip.MOV --preset m960d --dry-run                  # payloads to disk only
```

Try it against the mock:
```
python tests\mock_lastscene.py --port 8765 --save-dir data\mock_lastscene                 # terminal 1
```
In terminal 2, set the variables printed by the mock (the mock's own test credentials), then run `lastscene_bridge.run`.

## 5. Decisions for Dylan

1. **Detection confidence vs. memory confidence.**
   - **The conflict:** the contract says send detections ≥ 0.35, and treat < 0.5 as "maybe". Real keys mostly score **0.05–0.35** with our detector; the uploaded key memory above is 0.054.
   - **Share of keys-visible frames passing each floor** (our eval clips, `eval/`):

     | preset | placed on chair | cluttered table | walking back, close | ms/frame |
     |---|---|---|---|---|
     | `m960d` @0.35 | **1/22 (4%)** | **0/92** | **0/38** | 16 |
     | `m960d` @0.05 | 81% | 6% | 39% | 16 |
     | `x960d` @0.35 | 9/22 (40%) | 0/92 | 0/38 | 29 |

   - **What the 0.35 floor would cost:** it would drop nearly every real key memory. We **don't** apply it; `--enforce-contract-min-confidence` exists if you insist.
   - **Why our objects are already filtered:** every object we send is a *confirmed memory* (≥ 2 detections over ≥ 0.5 s), not a single detection. With `m960d` that produced **0 false memories on the no-keys clip at 0.05**. With our old default (`efficient`) the bridge *would* upload one false "keys" observation there (a laptop keyboard), so we'll ship `m960d` or better, pending ESP32 footage.
   - **Proposal:** accept our objects at any confidence, and show "maybe" below about 0.1. Or allow an optional `memory_confidence` field we'd add (confirmation strength), keeping `confidence` = detector confidence as the contract defines it.
2. **What "found it" does to memories.**
   - **Our side:** `bridge.found("keys")` deletes **every** local memory of that object (images, crop, records, event history), pauses new key memories for 5 s (they're in your hand), then calls `POST /api/v1/items/keys/found` with `found_at` and the last `observation_id` we sent.
   - **Questions:**
     - Should the backend keep the auto observations until the 72 h purge (history and Insights), or delete the item's frames on found, for privacy?
     - The contract's last-seen retention ends on found; we assume that.
     - `DELETE /observations/{id}` is for false detections only; we expose it, but nothing calls it automatically.
3. **Is `scene.id` stable enough to use as a room id? No, so we omit `scene`.**
   - ORB gives "same / different / unknown compared with *that* earlier memory". It doesn't assign persistent place ids or human labels.
   - On identical chairs it recognized the remembered spot in only 0–4 of 38 frames.
   - **Proposal:** a per-household mapping file, or Wi-Fi places. If useful later, we can send a non-semantic cluster id, but never as a room name.
4. **Projected historical location vs. live detection.**
   - We upload **only real detector boxes from the frame that was actually captured.** Projected boxes never leave the pipeline host.
   - `guide()` marks them separately:
     - `remembered_box_in_view` + `status: remembered_spot_in_view` means "they were here" (projected from the old frame);
     - `detection.visible` + `object_*` statuses means "they're here now".
   - On every evaluated frame, "visible now" was claimed only with a live detection.
   - **Ask:** on the dashboard, show the last-seen frame as historical (with `captured_at`). If guidance is ever shown there, draw "was here" differently from "is here".
5. **Smaller items to confirm:**
   - `pipeline.name`: we send `"lastseen-ml"`; the contract example says `"lastscene-vision"`. Which do you want?
   - Will your validator reject unknown fields? We send none.
   - `captured_at` for clips is accurate to within the clip length, since Apple's metadata may mark the start or the end of recording.
   - Our evidence frames are 1920×1080 JPEG, about 370 KB.
   - Where it runs: on a laptop host, with the ESP32 streaming to it, so credentials live on the laptop.

## 6. What was verified (Oct 17, local mock only)

- **Unit and integration tests, 19, against the mock:**
  - contract conversion; normalized boxes from uploaded dimensions (including after a downscale);
  - UUID stability; ISO UTC timestamps; no upload without a real capture time;
  - the confidence threshold reported, not dropped; event selection; batching;
  - the 2 s contract rate limit (measured server-side gap ≥ 2.0 s);
  - retry after a 500 with the same id; duplicate → 200; 429 handling;
  - the offline queue surviving a "restart" and flushing in `captured_at` order;
  - token refresh on 401; 422 → `failed/`; 413 → re-encode; redacted secrets;
  - `mvhd` capture-time parsing.
- **Real clips with the real detector (gated tests, plus a manual run):**
  - clip 01 → 1 observation, `201`;
  - the mock's last-seen for `keys` has `captured_at` 05:36:00.003Z (recording time + 6.0 s), and the box lands on the keys;
  - the no-keys clip → 0 observations with `m960d`; with `efficient` it would upload 1 false one.
- **Not verified:** Dylan's real endpoints, a real Auth0 tenant, and the ESP32 stream.
