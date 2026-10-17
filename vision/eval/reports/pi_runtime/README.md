# Pi runtime optimization, 2026-10-17

Measured by SSH on the actual Pi Zero 2 W (`aarch64`, 415 MiB usable RAM).
Model: `d26f737f1ea009a7`, YOLO11s INT8, 320 input, class `pin`.
Model SHA256: `0b73d633614bd3d82cf375d2918d9a5ed22bd9db7e04bbc8c41c84342da2de2a`.
Python 3.13.5, ONNX Runtime 1.31.0, OpenCV 4.10.0.

## What was actually running

No ML process was active at the initial inspection. The existing demo is
`~/live/live_feed.py`: a preview only, without memory creation or uploads.
It uses this model, three inference threads, low-memory mode, a 0.15 display
floor, and green boxes at 0.35. It requests 640x480 camera frames and encodes
the browser preview at JPEG quality 70. The HUD's `mem` number is **available
system memory**, not the model's RSS.

The USB camera advertises MJPEG at 1920x1080/30 fps and YUYV at 1920x1080/5 fps.
Our explicit MJPEG capture returned a genuine 1920x1080 array. Warm reads in
the first short probe took 0.052-0.061 s; this is not a sustained 30 fps result.
No focus control was exposed by `v4l2-ctl --list-ctrls`. Frame array dimensions
alone do not prove optical resolving power.

## Fixed-input runtime comparison

Separate processes, same model and original camera frame, one warmup, ten
measured `detect()` calls paced at one call per second. Measurements include
letterbox, inference and decode, but exclude live camera capture, browser
encoding, memory creation, upload and speech. All runs used low-memory mode
and the Pi's existing no-spin setting. CPU 100% means one logical core.

| Frame / setting | Median seconds | p95 seconds | CPU % | Peak RSS MiB |
|---|---:|---:|---:|---:|
| Initial frame, 3 ORT / 4 OpenCV threads | 0.734 | 0.749 | 181.7 | 176.0 |
| Initial frame, 4 ORT / 4 OpenCV threads | 0.621 | 0.656 | 189.2 | 193.6 |
| Initial frame, 4 ORT / 1 OpenCV thread | 0.620 | 0.653 | 189.3 | 206.8 |
| Clear pin frame, 3 ORT / 4 OpenCV threads | 0.739 | 0.846 | 182.3 | 205.0 |
| Clear pin frame, 4 ORT / 4 OpenCV threads | 0.623 | 0.711 | 189.6 | 198.8 |

Four ORT threads reduced median latency by about 15-16%. CPU time increased
slightly; this is not a power-saving claim. Effective cadence remained 1 fps
by design. Memory varied between processes; no memory improvement is claimed.
Changing OpenCV's thread count provided no useful gain and was not adopted.
Raw prediction hashes were identical between settings on each matched frame.
This is numerical equivalence on two images, not an accuracy evaluation.

The highest before/after temperature sample was 48.85 C. Swap was already in
use before the benchmark; swap activity during inference was not measured.
There was no observed OOM. Sustained camera + preview + speech contention
and power consumption remain unmeasured.

## Detection failure retained in the results

The initial frame had no visible pin. The first pin capture was blurred and
partially cut off. The later `pin-centered.png` capture clearly contains the
pin, but neither thread setting accepted it at 0.35. Full-frame maximum raw
class score was 0.000632. Existing full-frame + four overlapping tiles also
returned no detections and took 3.146 s on this one frame.

A human-selected 650x650 crop around the visible pin reached only 0.00438;
rotating the full frame 180 degrees reached 0.000175. Neither passed 0.35.
The manual crop is a diagnostic, not an automatic localization result.
This view cannot be fixed by choosing 0.05 instead of 0.35. It demonstrates
a detector failure on this view; these few samples cannot isolate lighting,
viewpoint, camera quality, training coverage or model/export limitations.
No recall, distance range or false-memory rate is established here.

## Applied and prepared changes

- Pi ML preview: changed only `threads=3` to `threads=4` in
  `/home/faker/live/live_feed.py`. Backup:
  `/home/faker/live/live_feed.py.before-codex-threads4`.
  The change takes effect on its next launch; no service was restarted.
- Local `lastscene_pipeline/runtime.py`: captured the no-spin setting already
  present in the Pi's runtime so it will survive a future code deployment.
  This was an existing device optimization, not a new discovery in this run.
- Added `scripts/bench_pi_runtime.py` for repeatable measurements and a real
  tiny-ONNX regression test checking low-memory behavior and output parity.
- Full local vision suite: **202 passed, 8 skipped**. The new test accounts for
  one additional pass over the preceding 201/8 run.

No model weights or confidence thresholds changed. Tiling remains outside
automatic memories. Backend/device client code was not changed. Images and
prediction arrays stay in ignored/private storage; only numeric JSON reports
are included here.

## Reproduce on the Pi

From the `vision/` directory, using a saved original camera image:

```bash
~/lastscene-venv/bin/python scripts/bench_pi_runtime.py \
  --model ~/lastscene-data/models/current --image /path/to/frame.png \
  --threads 3 --runs 10 --interval 1 --out /tmp/pi-3.json
~/lastscene-venv/bin/python scripts/bench_pi_runtime.py \
  --model ~/lastscene-data/models/current --image /path/to/frame.png \
  --threads 4 --runs 10 --interval 1 --out /tmp/pi-4.json
```

Run sequentially. The script also writes raw outputs beside each report as
`.npy`; compare hashes and retain these outside Git. The exact private images
from this run are identified by SHA256 in the JSON reports.

Start the optimized preview using the existing command:

```bash
~/lastscene-venv/bin/python ~/live/live_feed.py
```

The preview is at `http://100.89.232.47:8080`. It does not upload memories.
Restore the original preview if combined camera/speech testing regresses:

```bash
cp ~/live/live_feed.py.before-codex-threads4 ~/live/live_feed.py
```

Next: obtain fresh near/far/empty camera sequences, verify actual detections
and memory creation, then measure with preview and speech active together.
Keep model training/export experiments separate until this comparison exists.

## Retrained pin model `856455a8fa3c9474` (same evening)

Teach items had 22 pin photos by then: the original 10 close-ups plus 12 phone photos with the pin far
away (0.1–1.7% of the frame) on carpet, a chair and a cluttered desk. Trained with master's
`lastscene_pipeline.vultr run` from the Pi: plan `vc2-4c-8gb` (Atlanta, CPU), 13 min, about $0.06, server
destroyed afterwards. YOLO11s at 320 px, 40 COCO128 backgrounds, early stop at 51 epochs; map50 0.91 on its
own training photos (optimistic). INT8 matched 82% of FP32's detections on those photos (the old model: 100%).

Held-out check on the Pi (INT8, 2 threads, 0.35 floor, `live_feed.py` running, so latency is inflated):
2 earlier Pi frames with the pin (`pin-centered.png` clear; `pin.png` blurred and cut off) and 41 without
it (`camera.png` plus the 40 `calib/` frames). Those 40 were the old model's training backgrounds, so its
zero false alarms there are not a held-out result; the new model never saw them.

| Model, mode | Clear pin | Blurred pin | Empty frames with a hit |
|---|---:|---:|---:|
| old `d26f737f`, full frame | 0.0006 | 0.010 | 0/41 (seen in training) |
| new `856455a8`, full frame | **0.56** | 0.046 | 1/41 |
| new, 640x480 proxy | 0.57 | 0.071 | 0/41 |
| new, full frame + 4 tiles | 0.74 | 0.11 | **29/41** |

The one full-frame false alarm is a glowing exit sign (0.435, one frame; the neighbouring frames of the
same sign stayed below 0.2). Tiling makes the new model fire on arms and keyboards: keep it off. Two
positive frames are not a recall estimate; capture the near/mid/far/empty set below before relying on it.
The Pi's `models/current` now names `856455a8fa3c9474` (via `fetch`); roll back with
`echo d26f737f1ea009a7 > ~/lastscene-data/models/current`. `live_feed.py` picks it up on its next start.

On the Pi's hotspot, `api.vultr.com` is reached over IPv6, which Vultr's API access control rejected
although the IPv4 address was allowed; the runner was started through a wrapper that sets
`urllib3.util.connection.HAS_IPV6 = False`.

A first capture attempt (`pi_eval/20261017T222918Z`) saved nothing while
`live_feed.py` was running. Stop the preview first, since it owns the camera,
then capture 20 labeled frames and score them (frames stay in ignored storage):

```bash
~/lastscene-venv/bin/python scripts/capture_pi_eval.py
~/lastscene-venv/bin/python scripts/eval_pi_detection.py \
  --model ~/lastscene-data/models/856455a8fa3c9474 \
  --manifest ~/lastscene-data/pi_eval/<run id>/manifest.json --out /tmp/pi-eval.json
```
