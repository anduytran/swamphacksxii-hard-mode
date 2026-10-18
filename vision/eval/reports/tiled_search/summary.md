# Tiled search benchmark (exploratory)

Date: 2026-10-17. Machine: laptop with RTX 4060, Torch 2.11.0+cu128,
Ultralytics 8.4.178. This is not a measurement of the Pi or the live webcam.

## Protocol

The three modes ran in separate processes on the same original clip set, sampled
sequentially at 2 Hz, with prompt `bunch of keys`, confidence floor 0.05, and no
pre-inference resize. `m960d` is YOLOv8m-worldv2 at 960; `x960d` is YOLOv8x-worldv2
at 960; tiled search is the same XL model on the full frame plus four 2×2 crops
with 25% overlap. Tile boxes map back to source-frame pixel coordinates.

| Mode | Confirmed near | Labeled far | Keys in clutter | No-keys false frames | False memories | Median infer ms (near / far / clutter / no-keys) |
|---|---:|---:|---:|---:|---:|---:|
| m960d | 3/4 | 0/3 | 0/14 | 0/64 | 0 | 15.7 / 14.9 / 13.0 / 16.3 |
| x960d | 4/4 | 0/3 | 0/14 | 0/64 | 0 | 34.7 / 27.4 / 27.9 / 31.5 |
| x960d tiled | 4/4 | 2/3 | 14/14 | 8/64 | **1** | 143.9 / 134.8 / 133.8 / 170.4 |

Recall and false-frame counts use the 0.05 detector threshold. “No keys” is the
current manifest label for `IMG_7688.MOV`, but the manifest still lists visual
confirmation as pending, so those false-alarm counts remain provisional. Under
the production-style 2 Hz memory confirmation rules, tiled XL created a memory
at 21.52 s in that clip. Tiled XL increased far and clutter detections on these
clips, but it also created a false memory. Do not feed tiled proposals into
automatic capture or uploads based on this result.

Tiled XL latency was 133.8–170.4 ms median by scene group (p95 was 182.5 ms on
the no-keys clip), or 4.73 frames/s over the sequential offline pass. At 2 Hz,
that fits the laptop's 500 ms frame interval. Mean CPU use was 332.6% (roughly
3.3 logical cores); mean/max GPU utilization was 63.7%/90%; max GPU temperature
was 71 °C. Peak process RSS was 2,306 MB and peak CUDA allocation was 280 MB.
Those memory figures include Python, Torch, and the runtime, not just model
weights. They are host measurements only.

## Data limits

- `desk_a.mov`: 360×640, 5.338 s, 59.0 fps. Its old label window says far through
  9.9 s, but only three 2 Hz samples fall in the available far segment. The
  resulting 2/3 tiled recall is exploratory and too small to generalize.
- `desk_b.mov`: 360×640, 6.660 s, 59.0 fps. The old label window runs through
  9.9 s; only 14 samples exist in the keys-visible interval.
- `04_return_and_find.MOV`: 1920×1080, 11.625 s, 60 fps. The first 4.3 s are
  explicitly unscored because key visibility is unconfirmed. All three modes
  returned 0/9 proposals there; this is not a recall result.
- `IMG_7688.MOV`: 1080×1920, 31.638 s, 60 fps, manifest-labeled no-keys (visual
  confirmation pending). Tiled XL returned keys in 8/64 sampled frames and
  created one memory; both full-frame modes had 0/64 and created none.
- `IMG_7687.MP4` is referenced by an enrollment manifest but is not present in
  this clip directory. No live recording from `172.20.10.4:8080` was available.

## Decision

Keep `m960d` and the automatic-memory path as-is. Expose tiles only as an explicit
one-frame search: press `s` or type `search` in the host bridge. The current-frame
proposal is drawn as an orange dashed box and is neither stored nor uploaded.
Collect and label a fresh far/near/empty sequence from the actual camera stream
before considering continuous tiled inference.

## Reproduce

Run each mode separately so peak process memory is measured independently:

```powershell
cd vision
& ..\..\.venv\Scripts\python.exe scripts/bench_tiled_search.py --only m960d --out eval/reports/tiled_search/m960d.json
& ..\..\.venv\Scripts\python.exe scripts/bench_tiled_search.py --only x960d --out eval/reports/tiled_search/x960d.json
& ..\..\.venv\Scripts\python.exe scripts/bench_tiled_search.py --only x960d_tiled --out eval/reports/tiled_search/x960d_tiled.json
```
