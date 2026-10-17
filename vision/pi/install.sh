#!/usr/bin/env bash
# Raspberry Pi (Zero 2 W) runtime for lastscene_pipeline.detect / bench: ONNX Runtime + OpenCV, no torch.
# Needs 64-bit Raspberry Pi OS (Bookworm): onnxruntime has no 32-bit ARM wheels. Check with: uname -m  (aarch64)
set -euo pipefail

if [ "$(uname -m)" != "aarch64" ]; then
  echo "This needs 64-bit Raspberry Pi OS (uname -m says $(uname -m))." >&2
  exit 1
fi

# OpenCV, numpy, requests and picamera2 from apt: prebuilt for the Pi (pip would compile OpenCV for hours)
sudo apt-get update
sudo apt-get install -y python3-venv python3-opencv python3-numpy python3-requests python3-picamera2

VENV="${VENV:-$HOME/lastscene-venv}"
python3 -m venv --system-site-packages "$VENV"
"$VENV/bin/pip" install --upgrade pip
# numpy<2 keeps apt's numpy (apt's OpenCV is built against it)
"$VENV/bin/pip" install onnxruntime "numpy<2"
"$VENV/bin/python" -c "import onnxruntime, cv2, numpy; print('onnxruntime', onnxruntime.__version__, '| opencv', cv2.__version__, '| numpy', numpy.__version__)"
echo "Done. Next: copy a model folder and run the benchmark (pi/README.md)."
