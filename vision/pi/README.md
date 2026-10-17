# Running detect on the Raspberry Pi Zero 2 W

The laptop does `sync` and `train` (the GPU, Ultralytics, and the INT8 export). The Pi only runs `detect` with
`model-int8.onnx` (about 10 MB for YOLO11s), using ONNX Runtime and OpenCV, with no torch.
**Pi speed and memory are unmeasured:** run step 3 before relying on the Pi.

## 1. Install (once, on the Pi)

```bash
git clone -b ml https://github.com/anduytran/swamphacksxii-hard-mode.git
cd swamphacksxii-hard-mode/vision
bash pi/install.sh            # 64-bit Raspberry Pi OS only
```

## 2. Copy a model from the laptop

The Pi needs four files from `lastscene-data/models/<version>/` on the laptop: `model-int8.onnx`, `model.json`,
`classes.json` and `metrics.json`. It doesn't need `best.pt` or `model-fp32.onnx`.

```bash
# on the Pi
mkdir -p ~/lastscene-data/models/<version>
# copy the four files into it (scp from the laptop, or via the device you SSH from), then:
echo <version> > ~/lastscene-data/models/current
```

`detect` checks each file's sha256 against `model.json`, so a partial copy refuses to load.

## 3. Measure (report these numbers back)

```bash
cd ~/swamphacksxii-hard-mode/vision
rpicam-still -n -o /tmp/frame.jpg --width 1280 --height 720      # a real frame (optional)
~/lastscene-venv/bin/python -m lastscene_pipeline.bench --model ~/lastscene-data/models/current \
    --image /tmp/frame.jpg --runs 20 --threads 4 --low-memory
free -m; vcgencmd measure_temp
```

The output is one JSON line: `median_s`, `p95_s`, `peak_rss_mb`, `max_fps`. Usable means a median under 1 s
and peak RSS well under the 512 MB of RAM. If it's slower than that, retrain on the laptop at a smaller size
(`train --imgsz 320`; the size of a trained model is fixed) and copy the new folder.

## 4. Detect

```bash
sudo systemctl stop lastscene-device      # if Dylan's wearable client is running: it holds the camera
LASTSCENE_DATA=~/lastscene-data ~/lastscene-venv/bin/python -m lastscene_pipeline.detect \
    --source picam --fps 1 --scene-label "grand ballroom table" --env-file ~/.lastscene/pipeline.env
```

- Add `--dry-run` to print observations instead of posting them.
- `--fps` caps the inference rate. Set it at or below the measured `max_fps`; the memory thresholds scale with it.
- The env file holds `AUTH0_DOMAIN`, `AUTH0_AUDIENCE`, `DEVICE_CLIENT_ID`, `DEVICE_CLIENT_SECRET` and `PAIRING_CODE`. Keep it out of git.
- Low-memory mode (no ONNX Runtime arena) turns on by itself on boards with under 1 GB of RAM.

`--source picam` uses picamera2 and is **untested on hardware**. `--source http://<ip>:<port>/stream` (MJPEG)
and `--source 0` (V4L2) also work.
