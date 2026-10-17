# The "Last Scene" wake phrase

The wearable listens with [openWakeWord](https://github.com/dscripka/openWakeWord) (Apache-2.0), fully on the
Pi: audio never leaves it until the phrase is heard. "Last Scene" isn't one of openWakeWord's ready-made
models, so `last_scene.onnx` here (205 KB) is a custom one, trained from synthetic speech only (no
recordings of anyone). Device README step 4 copies this folder to `/opt/lastscene/wake/`, and
`device.env.example` already points `WAKE_WORD` at it.

## How it does (Oct 17)

Measured with `train/eval_wake.py` (max score per clip) and `train/fp_rate.py`:

| Test | Result |
|---|---|
| "last scene", 400 held-out synthetic voices | 99.5% wake at the 0.6 threshold (median score 0.99) |
| "Last Scene." in 6 macOS voices (never trained on) | 6/6 wake (0.93–0.99) |
| 10 everyday confusables ("last week", "the scene", "glass screen", "last of the scenes", ...) | 0/10 wake (highest 0.36) |
| 400 synthetic near-misses (phonetically close phrases) | 6.8% wake at 0.6 |
| ~11 h of general speech, music and noise | ~7 false wakes/hour at 0.6, ~5 at 0.8 (2 s cooldown) |
| "Hey Last Scene, where are my keys?" in one breath | 2/6 wake |

So: **say "Last Scene" on its own, wait for the beep, then speak.** Run into the next words, it often
misses. Synthetic voices aren't you: test with your own voice and the room you'll use before the demo.

"Last seen" sounds the same and will wake it; that's fine (the server treats both alike).

## Tune it

- Say "Last Scene" from a normal distance 10 times, and talk normally (including "last", "scene",
  "the last scene of the film") for a few minutes. Watch `journalctl -u lastscene-device -f`.
- Missed triggers: lower `WAKE_THRESHOLD` (e.g. 0.5). False triggers: raise it (0.7–0.8).
- In a loud room, the buttons are the reliable path. ~7 false wakes an hour means one every ~9 minutes
  of nonstop talk nearby; each one beeps and records what follows.

## Retrain it (about 1.5 h on a Mac CPU, no GPU)

`train/train_last_scene.py` runs the whole pipeline in stages, each skipped when its output exists, so a
rerun resumes. It mirrors openWakeWord's `train.py` with smaller downloads:

1. 3,000 + 400 "last scene" clips from [piper-sample-generator](https://github.com/rhasspy/piper-sample-generator)
   (LibriTTS-R, 904 speakers), plus as many phonetically close negatives ("last scene" and "last seen" are
   never used as negatives).
2. Augmentation: coloured noise and babble backgrounds, and 16 kHz mono room impulse responses in `rirs16k/`.
3. Negatives: four 2 GB slices spread across openWakeWord's 17 GB ACAV100M feature file (fetched with HTTP
   range requests, 8 GB on disk), and its false-alarm validation set (0.18 GB).
4. `auto_train`, 50,000 steps, then ONNX export. The weights are also saved as `out/last_scene.state.pt`;
   `EXPORT_ONLY=1` re-exports them without retraining.

Setup, in a work folder outside the repo (it fills ~12 GB): copy `train/*.py` there, clone
piper-sample-generator into it, put `libritts_r.pt` and its `.pt.json` in `models/`, add a few RIRs to `rirs16k/`, then

```bash
pip install torch openwakeword piper-tts==1.3.0 speechbrain torch_audiomentations scipy==1.14.1 requests onnxscript
STEPS=50000 python train_last_scene.py
python eval_wake.py out/last_scene.onnx "held-out=pos_test/*.wav" "near-misses=neg_test/*.wav"
python fp_rate.py out/last_scene.onnx
```

Notes from the first run: `scipy` 1.15+ breaks `torch_audiomentations` (`sph_harm`), so pin 1.14.1. torchaudio's
loaders now need torchcodec, so the script reads WAVs with SciPy instead. `gen.py` links espeak's data to
`~/.lastscene/espeak-ng-data`, because the library truncates long paths. With only the first 2 GB slice of
ACAV100M, false wakes ran 9–70/hour, so keep the four spread-out slices.

To try a different phrase (e.g. `hey last scene`, harder to trigger by accident), change `PHRASE` and delete
the `raw_*`, `pos_*`, `neg_*` folders and `out/*features*`. `WAKE_WORD=hey_jarvis` is a ready-made stand-in.
