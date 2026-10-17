"""Score the trained model: max score per clip (openWakeWord, 80 ms frames, 1 s of silence padding)."""
import sys, wave
from pathlib import Path
import numpy as np
from openwakeword.model import Model

model = Model(wakeword_models=[sys.argv[1]], inference_framework="onnx")
name = next(iter(model.models))

def score(path):
    with wave.open(str(path)) as w:
        pcm = np.frombuffer(w.readframes(w.getnframes()), np.int16)
    pad = np.zeros(16000, np.int16)
    x = np.concatenate([pad, pcm, pad])
    model.reset()
    return max(model.predict(x[i:i + 1280])[name] for i in range(0, len(x) - 1280, 1280))

for group in sys.argv[2:]:
    label, pattern = group.split("=")
    files = sorted(Path().glob(pattern))
    s = np.array([score(f) for f in files])
    print(f"{label:28s} n={len(s):4d}  >=0.5: {np.mean(s >= 0.5):5.1%}  >=0.6: {np.mean(s >= 0.6):5.1%}  median {np.median(s):.2f}")
    if len(files) <= 20:
        print("   ", ", ".join(f"{f.stem}:{v:.2f}" for f, v in zip(files, s)))
