#!/usr/bin/env bash
# Runs ON the rented GPU server: set up, train the "Last Scene" wake model, evaluate old vs new.
set -uo pipefail
export DEBIAN_FRONTEND=noninteractive
W=/root/wake
cd "$W"
step() { echo "== $(date -u +%H:%M:%S) $*"; }

step "apt"
apt-get update -q >/dev/null && apt-get install -y -q python3-venv python3-dev git curl build-essential >/dev/null
if lspci 2>/dev/null | grep -qi nvidia && ! nvidia-smi >/dev/null 2>&1; then
  step "installing the NVIDIA driver"
  apt-get install -y -q ubuntu-drivers-common >/dev/null
  drv=$(ubuntu-drivers devices 2>/dev/null | awk '/recommended/ {print $3; exit}')
  apt-get install -y -q "${drv:-nvidia-driver-570-server}" >/dev/null 2>&1 || true
  modprobe nvidia 2>/dev/null || true
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "NO GPU: training on $(nproc) CPUs"

step "python packages"
python3 -m venv venv
venv/bin/pip install -q --upgrade pip
venv/bin/pip install -q torch torchaudio --index-url https://download.pytorch.org/whl/cpu 2>&1 | tail -1
venv/bin/pip install -q piper-tts==1.3.0 speechbrain torch_audiomentations scipy==1.14.1 requests onnxscript onnx \
  onnxruntime scikit-learn tqdm mutagen torchinfo "torchmetrics<1" audiomentations acoustics pronouncing 2>&1 | tail -2
venv/bin/pip install -q --no-deps openwakeword==0.6.0 2>&1 | tail -1  # its Linux-only tflite-runtime has no wheel here
step "preflight"
venv/bin/python - <<'PY' || { step "FAILED preflight"; exit 1; }
import importlib
for m in ["openwakeword.data", "openwakeword.train", "openwakeword.utils", "torch_audiomentations", "speechbrain", "piper"]:
    importlib.import_module(m)
from openwakeword.data import generate_adversarial_texts
print("preflight ok:", generate_adversarial_texts(input_text="last scene", N=3, include_partial_phrase=1.0, include_input_words=0.2))
PY
venv/bin/python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
venv/bin/python -c "import openwakeword.utils as u; u.download_models()" 2>&1 | tail -1

step "voices and rooms"
git clone -q --depth 1 https://github.com/rhasspy/piper-sample-generator.git
mkdir -p models rirs16k
curl -sSL -o models/libritts_r.pt https://github.com/rhasspy/piper-sample-generator/releases/download/v2.0.0/en_US-libritts_r-medium.pt
curl -sSL -o models/libritts_r.pt.json https://huggingface.co/datasets/rhasspy/piper-checkpoints/resolve/main/en/en_US/libritts_r/medium/config.json
venv/bin/python - <<'EOF'
import numpy as np, scipy.io.wavfile as wf
from scipy.signal import resample_poly
from pathlib import Path
import train_last_scene as t
src = Path("piper-sample-generator/piper_sample_generator/impulses"); dst = Path("rirs16k")
for name in ["Concrete Room.wav", "ir_bathroom1.wav", "Accoustic2_Impulse.wav", "Blatty Plate.wav"]:
    p = src / name
    if p.exists():
        x, sr = t._wav_load(p)
        y = resample_poly(x.numpy().mean(axis=0), 16000, sr) if sr != 16000 else x.numpy().mean(axis=0)
        wf.write(dst / (p.stem.replace(" ", "_") + ".wav"), 16000, (y / (np.abs(y).max() + 1e-9) * 32767).astype(np.int16))
print("RIRs:", sorted(p.name for p in dst.glob("*.wav")))
EOF

step "training"
PYTHONUNBUFFERED=1 STEPS=50000 venv/bin/python train_last_scene.py || { step "FAILED training"; exit 1; }

step "evaluating"
for m in old_last_scene.onnx out/last_scene.onnx; do
  echo "-- $m"
  venv/bin/python eval_wake.py "$m" "held-out=pos_test/*.wav" "near-misses=neg_test/*.wav" "lookalikes=hard_test/*.wav"
  venv/bin/python fp_rate.py "$m"
done
step "DONE"
