"""False wakes per hour on openWakeWord's validation audio (~11 h of speech, music, noise), counted the
way the device does: a score >= threshold triggers, then 2 s (25 frames of 80 ms) of cooldown."""
import sys
import numpy as np
import onnxruntime as ort

sess = ort.InferenceSession(sys.argv[1])
inp = sess.get_inputs()[0]
n = inp.shape[1]
feats = np.load("out/validation_set_features.npy").astype(np.float32)
win = np.lib.stride_tricks.sliding_window_view(feats, (n, feats.shape[1]))[:, 0]
scores = np.array([sess.run(None, {inp.name: win[i:i + 1]})[0].ravel()[0] for i in range(len(win))])
hours = len(feats) * 0.08 / 3600
for th in (0.5, 0.6, 0.7, 0.8, 0.9):
    wakes, cool = 0, 0
    for s in scores:
        if cool:
            cool -= 1
        elif s >= th:
            wakes, cool = wakes + 1, 25
    print(f"threshold {th}: {wakes} wakes in {hours:.1f} h = {wakes / hours:.2f}/hour")
