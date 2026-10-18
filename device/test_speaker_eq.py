"""Speaker level and EQ: A4 and B4 come out 13 dB under the rest, AUDIO_GAIN_DB moves everything,
and streamed chunks sound the same as one block."""

import array
import io
import math
import time
import wave

import play_scale
import speaker_eq


def level_db(pcm: bytes) -> float:
    s = array.array("h", pcm)
    return 20 * math.log10(math.sqrt(sum(x * x for x in s) / len(s)))


def note_levels(eq: speaker_eq.SpeakerEq) -> dict[float, float]:
    """Each scale note on its own (steady part only), before and after the EQ: the change in dB."""
    out = {}
    for f in play_scale.NOTES:
        play_scale.NOTES, saved = [f], play_scale.NOTES
        try:
            pcm = play_scale.scale_pcm(note_seconds=0.5, gap_seconds=0)
        finally:
            play_scale.NOTES = saved
        mid = slice(len(pcm) // 2 // 2 * 2, len(pcm) * 9 // 10 // 2 * 2)
        e = speaker_eq.SpeakerEq(play_scale.RATE, 0, []) if eq is None else eq
        e.state = [[0.0] * 4 for _ in e.coeffs]
        out[round(f)] = level_db(e.process(pcm)[mid]) - level_db(pcm[mid])
    return out


def test_default_eq_takes_a_and_b_13_db_under_the_rest():
    eq = speaker_eq.from_env(play_scale.RATE, {})
    levels = note_levels(eq)
    assert abs(levels[440] + 13) < 0.5 and abs(levels[494] + 13) < 0.5
    for f in (262, 294, 330, 349, 392):
        assert levels[f] > -1.5, (f, levels[f])
    assert levels[523] > -4


def test_gain_moves_everything_and_empty_eq_is_flat():
    eq = speaker_eq.from_env(play_scale.RATE, {"AUDIO_GAIN_DB": "-6", "SPEAKER_EQ": ""})
    assert all(abs(v + 6) < 0.1 for v in note_levels(eq).values())
    assert speaker_eq.from_env(24000, {"SPEAKER_EQ": ""}).flat


def test_chunks_sound_like_one_block():
    pcm = play_scale.scale_pcm()
    whole = speaker_eq.from_env(play_scale.RATE, {}).process(pcm)
    eq = speaker_eq.from_env(play_scale.RATE, {})
    chunked = b"".join(eq.process(pcm[i:i + 4800]) for i in range(0, len(pcm), 4800))
    assert chunked == whole


def test_wav_keeps_its_format_and_other_formats_pass_through():
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1), w.setsampwidth(2), w.setframerate(16000)
        w.writeframes(array.array("h", (int(8000 * math.sin(2 * math.pi * 440 * i / 16000)) for i in range(8000))).tobytes())
    out = speaker_eq.apply_wav(buf.getvalue(), {})
    with wave.open(io.BytesIO(out)) as w:
        assert (w.getframerate(), w.getnchannels(), w.getnframes()) == (16000, 1, 8000)
    assert speaker_eq.apply_wav(b"not a wav", {}) == b"not a wav"


def test_loud_gain_clips_instead_of_wrapping():
    pcm = array.array("h", [30000, -30000] * 100).tobytes()
    out = array.array("h", speaker_eq.SpeakerEq(24000, 12, []).process(pcm))
    assert max(out) == 32767 and min(out) == -32768


def test_five_seconds_of_speech_is_quick():
    pcm = play_scale.tone_pcm(seconds=5.0)
    t = time.perf_counter()
    speaker_eq.from_env(24000, {}).process(pcm)
    assert time.perf_counter() - t < 1.0  # a laptop; a Pi Zero 2 W is several times slower
