"""Wrapper: point piper's espeak data at a short path (its C library truncates long paths), then generate."""
import os, sys
from pathlib import Path
SHORT = Path.home() / ".lastscene" / "espeak-ng-data"
import piper.phonemize_espeak as pe
if not SHORT.exists():  # link to the espeak data that ships with piper-tts
    SHORT.parent.mkdir(parents=True, exist_ok=True)
    SHORT.symlink_to(Path(pe.ESPEAK_DATA_DIR))
pe.ESPEAK_DATA_DIR = SHORT
pe.EspeakPhonemizer.__init__.__defaults__ = (SHORT,)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "piper-sample-generator"))
from piper_sample_generator.__main__ import generate_samples  # noqa: E402
