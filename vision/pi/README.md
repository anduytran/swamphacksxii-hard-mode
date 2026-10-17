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

### 2b. No trained model yet: export YOLO-World with a fixed vocabulary (on the laptop)

`set_classes` runs YOLO-World's CLIP text encoder once on the laptop and stores the class embeddings in the
model, so the exported ONNX is a plain detector and the Pi never needs CLIP. The folder has the same layout
as a trained one, so `bench` and `detect` load it unchanged. Calibrate on frames from the Pi's own camera.

```bash
# laptop, in vision/, with ultralytics + onnx + onnxruntime + onnxslim installed
python -m scripts.export_world_pi --classes keys wallet phone glasses remote --imgsz 320 \
    --calib path/to/pi-frames --out lastscene-data/models
rsync -a --exclude model-fp32.onnx lastscene-data/models/<version> faker:lastscene-data/models/
ssh faker 'echo <version> > ~/lastscene-data/models/current'
```

Measured on Oct 17, 2026 (Pi Zero 2 W, 415 MB usable RAM, Debian 13, Python 3.13, onnxruntime 1.31.0),
`yolov8s-worldv2` at 320 px, INT8 (13.5 MB), `--low-memory`, on a 1280x720 USB-camera frame:

| threads | load | median | p95 | max fps | peak RSS |
|---|---|---|---|---|---|
| 4 | 3.4 s | 0.82 s | 0.83 s | 1.2 | 201 MB |
| 2 | 3.1 s | 1.27 s | 1.35 s | 0.8 | 193 MB |

INT8 kept 9 of 10 FP32 detections on the 44 calibration frames (not a held-out accuracy test). Untrained
YOLO-World scores real keys low (README: about 0.05-0.35), so few sightings reach detect's default
`--min-confidence 0.35`; a model trained from the Teach items photos is still the better path.

The Pi in this setup uses a USB camera: use `--source 0`, not `--source picam`. A 90 s `detect --source 0 --fps 1 --threads 4
--dry-run` ran cleanly: frames arrive at 640x480 (the GStreamer warning is harmless, OpenCV falls back to
V4L2), RAM stayed under 150 MB used, and the board reached 70 °C with no throttling. Keep `--fps` at 1. Debian 13 (trixie) ships
numpy 2 with its OpenCV, so skip install.sh's `numpy<2` pin there (`pip install onnxruntime pytest` in the
venv is enough when the apt packages are already installed).

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
