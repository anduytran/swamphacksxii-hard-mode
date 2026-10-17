# The "Last Scene" wake phrase

The wearable listens with [openWakeWord](https://github.com/dscripka/openWakeWord) (Apache-2.0), fully on the
Pi: audio never leaves it until the phrase is heard. "Last Scene" isn't one of openWakeWord's ready-made
models, so it needs a small custom model, trained once from synthetic speech (no recordings of anyone).

## Train it (about an hour, once)

1. Open openWakeWord's **automatic model training** notebook (linked from its README) in Google Colab with
   a GPU runtime (free tier is enough).
2. Set the target phrase to `last scene` (lower case, as spoken), keep the defaults for the number of
   examples and steps, and run all cells. It generates thousands of synthetic "last scene" clips with
   different voices, mixes in noise and room sound, and trains against general speech so other words
   don't trigger it.
3. Download the result as `last_scene.onnx` and copy it to the Pi:
   ```bash
   scp last_scene.onnx <pi>:/tmp/ && ssh <pi> 'sudo mv /tmp/last_scene.onnx /opt/lastscene/models/'
   ```
4. In `/etc/lastscene/device.env`: `WAKE_WORD=/opt/lastscene/models/last_scene.onnx`, then
   `sudo systemctl restart lastscene-device`.

## Tune it

- Say "Last Scene" from a normal distance 10 times, and talk normally (including "last", "scene",
  "the last scene of the film") for a few minutes. Check `journalctl -u lastscene-device -f`.
- Missed triggers: lower `WAKE_THRESHOLD` (e.g. 0.5). False triggers: raise it (e.g. 0.7).
- Two short words are easier to trigger by accident than a longer phrase. If it fires too often in the
  ballroom, retrain with `hey last scene` and say that instead (the server already strips either).

Until the model exists, `WAKE_WORD=hey_jarvis` is a working stand-in: the same flow, with "Hey Jarvis".
