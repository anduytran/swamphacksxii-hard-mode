"""Speaker level and EQ for everything the wearable plays (beeps, spoken answers, the test scale).

AUDIO_GAIN_DB    one gain for all sound, in dB (default 0). Calibrate once with a phone sound meter:
                 `play_scale.py --tone` plays a steady 1 kHz tone at about the level of speech; set
                 AUDIO_GAIN_DB = target - reading (e.g. 80 dB wanted, 86 read -> -6). The demo runs at
                 -10: about 70 dB at most.
SPEAKER_EQ       cuts or boosts where the speaker is uneven, as freq:dB:Q bands separated by commas.
                 The default evens out the Voice Bonnet speaker, which plays A4 and B4 (440 and 494 Hz)
                 much louder than the rest: A4 -7 dB, B4 -10 dB (the two bands overlap, hence -6.2 and
                 -9.5), the rest of the scale within 1 dB, C5 -2.5 dB. Empty = flat.

No dependencies: plain Python biquads (RBJ peaking EQ); 5 s of speech takes 0.03 s on a laptop, an estimated ~0.5 s on a Pi Zero 2 W.
"""

import array
import io
import math
import os
import wave

DEFAULT_EQ = "440:-6.2:16,494:-9.5:16"


def parse_bands(spec: str) -> list[tuple[float, float, float]]:
    bands = []
    for part in (spec or "").split(","):
        if part.strip():
            f, g, q = (float(x) for x in part.split(":"))
            bands.append((f, g, q))
    return bands


def _peaking(f0: float, gain_db: float, q: float, rate: int) -> tuple[float, ...]:
    a = 10 ** (gain_db / 40)
    w = 2 * math.pi * f0 / rate
    alpha, cos = math.sin(w) / (2 * q), math.cos(w)
    a0 = 1 + alpha / a
    return ((1 + alpha * a) / a0, -2 * cos / a0, (1 - alpha * a) / a0, -2 * cos / a0, (1 - alpha / a) / a0)


class SpeakerEq:
    """Gain plus peaking bands on 16-bit mono PCM. Keeps filter state between calls, so a stream can be
    fed in chunks (Gemini Live) and sound the same as one block."""

    def __init__(self, rate: int, gain_db: float = 0.0, bands=()):
        self.gain = 10 ** (gain_db / 20)
        self.coeffs = [_peaking(f, g, q, rate) for f, g, q in bands if 0 < f < rate / 2]
        self.state = [[0.0, 0.0, 0.0, 0.0] for _ in self.coeffs]  # x1 x2 y1 y2

    @property
    def flat(self) -> bool:
        return not self.coeffs and self.gain == 1

    def process(self, pcm: bytes) -> bytes:
        if self.flat or not pcm:
            return pcm
        samples = array.array("h", pcm[: len(pcm) // 2 * 2])
        out = [x * self.gain for x in samples]
        for (b0, b1, b2, a1, a2), st in zip(self.coeffs, self.state):
            x1, x2, y1, y2 = st
            for i, x in enumerate(out):
                y = b0 * x + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
                x2, x1, y2, y1 = x1, x, y1, y
                out[i] = y
            st[:] = [x1, x2, y1, y2]
        return array.array("h", (32767 if y > 32767 else -32768 if y < -32768 else int(y) for y in out)).tobytes()


def from_env(rate: int, env=None) -> SpeakerEq:
    env = os.environ if env is None else env
    return SpeakerEq(rate, float(env.get("AUDIO_GAIN_DB") or 0), parse_bands(env.get("SPEAKER_EQ", DEFAULT_EQ)))


def apply_wav(data: bytes, env=None) -> bytes:
    """A WAV through the speaker EQ; anything that isn't 16-bit mono PCM WAV plays as it came."""
    try:
        with wave.open(io.BytesIO(data)) as w:
            if w.getsampwidth() != 2 or w.getnchannels() != 1:
                return data
            rate, pcm = w.getframerate(), w.readframes(w.getnframes())
    except (wave.Error, EOFError):
        return data
    eq = from_env(rate, env)
    if eq.flat:
        return data
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1), w.setsampwidth(2), w.setframerate(rate)
        w.writeframes(eq.process(pcm))
    return buf.getvalue()
