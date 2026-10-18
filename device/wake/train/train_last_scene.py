"""Train the "last scene" openWakeWord model on this Mac (CPU), stage by stage; each stage is skipped
if its output exists, so a rerun resumes. Mirrors openwakeword/train.py with smaller downloads."""

import logging
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
OUT = HERE / "out"
OUT.mkdir(exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", stream=sys.stdout)
log = logging.getLogger("train")
random.seed(0)
np.random.seed(0)

PHRASE = "last scene"
N_TRAIN, N_TEST = 3000, 400
FEATS_URL = "https://huggingface.co/datasets/davidscripka/openwakeword_features/resolve/main/"
SUBSET_BYTES = 2_000_000_000
ACAV_ROWS = 5_625_000  # rows of (16, 96) float16 in the full file (~17.3 GB); slices stay inside it
GEN_MODEL = str(HERE / "models" / "libritts_r.pt")


def stage(name, path):
    done = Path(path).exists() and (not Path(path).is_dir() or any(Path(path).iterdir()))
    log.info(("skip  " if done else "start ") + name)
    return not done


ACAV_SLICES = (0.0, 0.25, 0.5, 0.75)  # where each 2 GB slice starts in the 17 GB file: varied speech, music, noise


def download_subsets():
    return [download_subset(i, at) for i, at in enumerate(ACAV_SLICES)]


def download_subset(i, at):
    """~2 GB of the 17 GB ACAV100M feature file from `at` (a fraction of it), written as a valid .npy."""
    import requests

    dst = OUT / ("acav_subset.npy" if i == 0 else f"acav_subset_{i}.npy")
    if not stage(f"download general speech features (2 GB slice at {at:.0%} of 17 GB)", dst):
        return dst
    rows = (SUBSET_BYTES - 128) // 3072
    start = 128 + int(at * ACAV_ROWS) * 3072
    tmp = dst.with_suffix(".part")
    with requests.get(FEATS_URL + "openwakeword_features_ACAV100M_2000_hrs_16bit.npy",
                      headers={"Range": f"bytes={start}-{start + rows * 3072 - 1}"}, stream=True, timeout=60) as r:
        r.raise_for_status()
        arr = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float16, shape=(rows, 16, 96))
        flat = arr.reshape(-1).view(np.uint8)
        pos = 0
        for chunk in r.iter_content(1 << 22):
            flat[pos:pos + len(chunk)] = np.frombuffer(chunk, np.uint8)
            pos += len(chunk)
        arr.flush()
        assert pos == rows * 3072, (pos, rows * 3072)
    os.replace(tmp, dst)
    return dst


def download_validation():
    import requests

    dst = OUT / "validation_set_features.npy"
    if stage("download false-alarm validation set (0.18 GB)", dst):
        with requests.get(FEATS_URL + "validation_set_features.npy", stream=True, timeout=60) as r, open(dst.with_suffix(".part"), "wb") as f:
            r.raise_for_status()
            for chunk in r.iter_content(1 << 22):
                f.write(chunk)
        os.replace(dst.with_suffix(".part"), dst)
    return dst


def generate(texts, out_dir, n):
    import gen

    out_dir = Path(out_dir)
    have = len(list(out_dir.glob("*.wav"))) if out_dir.exists() else 0
    if have >= n:
        log.info(f"skip  {out_dir.name}: {have} clips")
        return
    log.info(f"start {out_dir.name}: {n - have} clips")
    gen.generate_samples(text=texts, output_dir=str(out_dir), model=GEN_MODEL, max_samples=n - have, batch_size=50,
                         noise_scales=[0.98], noise_scale_ws=[0.98], length_scales=[0.75, 1.0, 1.25], auto_reduce_batch_size=True,
                         file_names=[f"{have + i:05d}.wav" for i in range(n - have)])


def resample_dir(src, dst):
    """The generator writes 22.05 kHz; openWakeWord trains on 16 kHz."""
    import torchaudio

    src, dst = Path(src), Path(dst)
    dst.mkdir(exist_ok=True)
    todo = [p for p in sorted(src.glob("*.wav")) if not (dst / p.name).exists()]
    if not todo:
        return
    log.info(f"resample {len(todo)} clips -> {dst.name}")
    rs = torchaudio.transforms.Resample(22050, 16000)
    import scipy.io.wavfile as wf

    for p in todo:
        sr, data = wf.read(p)
        x = rs(__import__("torch").from_numpy(data.astype(np.float32) / 32768)[None])[0].numpy()
        wf.write(dst / p.name, 16000, (np.clip(x, -1, 1) * 32767).astype(np.int16))


# Heard in real use as "Last Scene" (Oct 18: "Let's see. Is my brother going?" woke it at 0.77). Taught as negatives
# many times over, and kept out of training in HARD_TEST to measure them.
HARD = ["let's see", "let me see", "let's see now", "let's see it", "let's see what", "let's see is my brother going",
        "let's see if it works", "okay let's see", "so let's see", "let's say", "let's eat", "let's sing", "let's seem",
        "lets see", "less seen", "lots of scenes", "last sea", "list scene", "lest we see", "let us see",
        "let's sit", "last thing", "let's leave", "best scene", "let's see here", "well let's see"]
HARD_TEST = ["let's see", "let's see is my brother going", "let me see", "let's say", "so let's see", "let's eat"]


def adversarial_texts(n):
    from openwakeword.data import generate_adversarial_texts

    texts = generate_adversarial_texts(input_text=PHRASE, N=n, include_partial_phrase=1.0, include_input_words=0.2)
    extra = ["last", "scene", "at last", "the last time", "last week", "fast", "past", "lasting", "the scene",
             "a scene", "seen it", "screen", "the last one", "last night", "at least", "less screen", "glass scene",
             "last season", "last sign", "lost scenery", "class seen", "vast screen", "last minute", "scenic"]
    # "last seen" sounds exactly like "last scene" (scene = seen): never teach it as a negative.
    bad = {"last scene", "last seen", "lastscene"}
    out = [t for t in texts + extra * 20 + HARD * 40
           if t.strip().lower() not in bad and "last scene" not in t.lower() and "last seen" not in t.lower()]
    random.shuffle(out)
    return out[:n] if len(out) >= n else (out * (n // max(1, len(out)) + 1))[:n]


def make_backgrounds(neg_dir):
    """No big noise corpus here: coloured noise at several levels, plus babble mixed from negative speech."""
    import scipy.io.wavfile as wf

    dst = OUT / "background"
    if not stage("background noise", dst):
        return dst
    dst.mkdir()
    sr, n = 16000, 16000 * 10
    for i in range(30):
        white = np.random.randn(n)
        kind = i % 3
        if kind == 1:  # pink-ish
            white = np.cumsum(white) - np.convolve(np.cumsum(white), np.ones(200) / 200, mode="same")
        elif kind == 2:  # brown-ish
            white = np.cumsum(white)
            white -= np.convolve(white, np.ones(2000) / 2000, mode="same")
        white = white / (np.abs(white).max() + 1e-9) * random.uniform(0.05, 0.5)
        wf.write(dst / f"noise_{i:02d}.wav", sr, (white * 32767).astype(np.int16))
    clips = sorted(Path(neg_dir).glob("*.wav"))
    for i in range(30):
        mix = np.zeros(n)
        for _ in range(8):
            _, c = wf.read(random.choice(clips))
            c = c.astype(np.float32) / 32768
            start = random.randint(0, max(0, n - len(c)))
            mix[start:start + len(c)] += c[: n - start] * random.uniform(0.2, 0.8)
        mix = mix / (np.abs(mix).max() + 1e-9) * random.uniform(0.1, 0.6)
        wf.write(dst / f"babble_{i:02d}.wav", sr, (mix * 32767).astype(np.int16))
    return dst


def _wav_info(path):
    """torchaudio.info, removed in newer torchaudio (torch_audiomentations still calls it)."""
    import types
    import scipy.io.wavfile as wf

    sr, data = wf.read(str(path), mmap=True)
    return types.SimpleNamespace(num_frames=data.shape[0], sample_rate=sr, num_channels=1 if data.ndim == 1 else data.shape[1])


def _wav_load(path, frame_offset=0, num_frames=-1, *args, **kwargs):
    """torchaudio.load without torchcodec (newer torchaudio needs it even for WAV): SciPy instead."""
    import torch
    import scipy.io.wavfile as wf

    sr, data = wf.read(str(path))
    data = data[frame_offset:] if num_frames is None or num_frames < 0 else data[frame_offset:frame_offset + num_frames]
    scale = {np.dtype(np.int16): 32768.0, np.dtype(np.int32): 2147483648.0, np.dtype(np.uint8): 128.0}.get(data.dtype, 1.0)
    x = (data.astype(np.float32) - (128.0 if data.dtype == np.uint8 else 0.0)) / scale
    x = x[None, :] if x.ndim == 1 else x.T
    return torch.from_numpy(np.ascontiguousarray(x)), sr


def features(clips_dir, out_file, total_length, backgrounds, rirs, rounds=1):
    import torchaudio
    from openwakeword.data import augment_clips
    from openwakeword.utils import compute_features_from_generator

    torchaudio.load, torchaudio.info = _wav_load, _wav_info
    if not stage(f"features {Path(out_file).name}", out_file):
        return
    clips = [str(p) for p in sorted(Path(clips_dir).glob("*.wav"))] * rounds
    gen_ = augment_clips(clips, total_length=total_length, batch_size=16, background_clip_paths=backgrounds, RIR_paths=rirs)
    part = Path(str(out_file).replace(".npy", ".part.npy"))
    compute_features_from_generator(gen_, n_total=len(clips), clip_duration=total_length, output_file=str(part),
                                    device="cpu", ncpu=max(1, (os.cpu_count() or 2) // 2))
    os.replace(part, out_file)  # only a finished file counts as done


def train(acav, valfp, total_length, steps):
    import torch
    import openwakeword
    from openwakeword.data import mmap_batch_generator
    from openwakeword.train import Model

    final = OUT / "last_scene.onnx"
    if not stage("train", final):
        return final
    F = openwakeword.utils.AudioFeatures(device="cpu")
    input_shape = F.get_embedding_shape(total_length // 16000)
    oww = Model(n_classes=1, input_shape=input_shape, model_type="dnn", layer_dim=32,
                seconds_per_example=1280 * input_shape[0] / 16000)

    def f(x, n=input_shape[0]):
        if n != x.shape[1]:
            x = np.vstack(x)
            return np.array([x[i:i + n, :] for i in range(0, x.shape[0] - n, n)])
        return x

    files = {**{f"ACAV100M_{i}": str(a) for i, a in enumerate(acav)}, "positive": str(OUT / "positive_features_train.npy"),
             "adversarial_negative": str(OUT / "negative_features_train.npy")}
    data_transforms = {k: f for k in files}
    label_transforms = {k: ((lambda x: [1 for _ in x]) if k == "positive" else (lambda x: [0 for _ in x])) for k in files}
    batches = mmap_batch_generator(files, n_per_class={**{f"ACAV100M_{i}": 384 for i in range(len(acav))}, "positive": 50, "adversarial_negative": 100},
                                   data_transform_funcs=data_transforms, label_transform_funcs=label_transforms)

    class It(torch.utils.data.IterableDataset):
        def __iter__(self):
            return batches

    X_train = torch.utils.data.DataLoader(It(), batch_size=None, num_workers=0)
    fp = np.load(valfp)
    fp = np.array([fp[i:i + input_shape[0]] for i in range(0, fp.shape[0] - input_shape[0], 1)])
    X_val_fp = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.from_numpy(fp), torch.zeros(len(fp))), batch_size=len(fp))
    pos, neg = np.load(OUT / "positive_features_test.npy"), np.load(OUT / "negative_features_test.npy")
    labels = np.hstack((np.ones(len(pos)), np.zeros(len(neg)))).astype(np.float32)
    X_val = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.from_numpy(np.vstack((pos, neg))), torch.from_numpy(labels)),
                                        batch_size=len(labels))
    best = oww.auto_train(X_train=X_train, X_val=X_val, false_positive_val_data=X_val_fp, steps=steps,
                          max_negative_weight=1500, target_fp_per_hour=0.2)
    torch.save(best.state_dict(), OUT / "last_scene.state.pt")  # keep the weights even if the export fails
    export(best, input_shape, final)
    return final


def export(model, input_shape, path):
    """ONNX via the classic exporter (torch's new default one needs onnxscript); same graph openWakeWord exports."""
    import copy

    import torch

    torch.onnx.export(copy.deepcopy(model).to("cpu").eval(), torch.rand(input_shape)[None, ], str(path),
                      opset_version=13, dynamo=False)


def export_saved():
    """Export out/last_scene.state.pt (written before the export) without retraining."""
    import openwakeword
    import torch
    from openwakeword.train import Model

    input_shape = openwakeword.utils.AudioFeatures(device="cpu").get_embedding_shape(32000 // 16000)
    oww = Model(n_classes=1, input_shape=input_shape, model_type="dnn", layer_dim=32,
                seconds_per_example=1280 * input_shape[0] / 16000)
    oww.model.load_state_dict(torch.load(OUT / "last_scene.state.pt"))
    export(oww.model, input_shape, OUT / "last_scene.onnx")
    log.info(f"DONE exported {OUT / 'last_scene.onnx'}")


def main():
    if os.environ.get("EXPORT_ONLY"):
        return export_saved()
    t0 = time.time()
    acav, valfp = download_subsets(), download_validation()
    generate(PHRASE, HERE / "raw_pos_train", N_TRAIN)
    generate(PHRASE, HERE / "raw_pos_test", N_TEST)
    adv = adversarial_texts(N_TRAIN + N_TEST)
    generate(adv[:N_TRAIN], HERE / "raw_neg_train", N_TRAIN)
    generate(adv[N_TRAIN:], HERE / "raw_neg_test", N_TEST)
    generate(HARD_TEST, HERE / "raw_hard_test", 120)  # for eval_wake.py: "lookalikes=hard_test/*.wav"
    for name in ("pos_train", "pos_test", "neg_train", "neg_test", "hard_test"):
        resample_dir(HERE / f"raw_{name}", HERE / name)
    import scipy.io.wavfile as wf

    lengths = [len(wf.read(p)[1]) for p in random.sample(sorted((HERE / "pos_test").glob("*.wav")), 50)]
    total_length = int(round(np.median(lengths) / 1000) * 1000) + 12000
    total_length = 32000 if total_length < 32000 or abs(total_length - 32000) <= 4000 else total_length
    log.info(f"clip length {total_length / 16000:.2f} s")
    backgrounds = [str(p) for p in make_backgrounds(HERE / "neg_train").glob("*.wav")]
    rirs = [str(p) for p in (HERE / "rirs16k").glob("*.wav")]  # real rooms only, 16 kHz mono (see the run notes)
    features(HERE / "pos_train", OUT / "positive_features_train.npy", total_length, backgrounds, rirs, rounds=2)
    features(HERE / "neg_train", OUT / "negative_features_train.npy", total_length, backgrounds, rirs, rounds=2)
    features(HERE / "pos_test", OUT / "positive_features_test.npy", total_length, backgrounds, rirs)
    features(HERE / "neg_test", OUT / "negative_features_test.npy", total_length, backgrounds, rirs)
    model = train(acav, valfp, total_length, steps=int(os.environ.get("STEPS", 20000)))
    log.info(f"DONE {model} in {(time.time() - t0) / 60:.0f} min")


if __name__ == "__main__":
    main()
