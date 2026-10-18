"""Speaker check: plays a C major scale (C4 to C5) through aplay.

  python3 play_scale.py                 # once, on AUDIO_OUT_DEVICE (or "default")
  python3 play_scale.py -D plughw:1,0   # a named ALSA device, e.g. the Voice Bonnet
  python3 play_scale.py --loop          # repeat until Ctrl-C
  python3 play_scale.py --tone          # steady 1 kHz at about speech level, to read a sound meter
  python3 play_scale.py --flat          # without AUDIO_GAIN_DB / SPEAKER_EQ (speaker_eq.py)

Uses the same raw 24 kHz mono stream the device sends Gemini Live replies through, so if this
plays, spoken answers will too.
"""
import argparse
import math
import os
import struct
import subprocess

try:
    import speaker_eq
except ImportError:
    speaker_eq = None

RATE = 24000
NOTES = [261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88, 523.25]  # C4 D4 E4 F4 G4 A4 B4 C5


def scale_pcm(note_seconds=0.3, gap_seconds=0.4, volume=9000):
    """16-bit mono PCM: each note with a short fade in/out so it doesn't click, then a gap."""
    out = bytearray()
    n = int(RATE * note_seconds)
    for f in NOTES:
        for i in range(n):
            env = min(1, i / 240, (n - i) / 1200)
            out += struct.pack("<h", int(volume * env * math.sin(2 * math.pi * f * i / RATE)))
    out += bytes(2 * int(RATE * gap_seconds))
    return bytes(out)


def tone_pcm(freq=1000, seconds=1.0, volume=3300):
    """A steady sine; amplitude 3300 is -20 dBFS RMS, about how loud speech is on average."""
    n = int(RATE * seconds)
    return b"".join(struct.pack("<h", int(volume * math.sin(2 * math.pi * freq * i / RATE))) for i in range(n))


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("-D", "--device", default=os.environ.get("AUDIO_OUT_DEVICE", "default"),
                   help="ALSA output device (default: $AUDIO_OUT_DEVICE or 'default')")
    p.add_argument("--loop", action="store_true", help="repeat until Ctrl-C")
    p.add_argument("--tone", action="store_true", help="steady 1 kHz tone at speech level until Ctrl-C")
    p.add_argument("--flat", action="store_true", help="skip AUDIO_GAIN_DB and SPEAKER_EQ")
    args = p.parse_args()

    if args.tone:
        args.loop = True
    pcm = tone_pcm() if args.tone else scale_pcm()
    if speaker_eq is not None and not args.flat:
        pcm = speaker_eq.from_env(RATE).process(pcm)  # the speaker's settings, as answers get them
    proc = subprocess.Popen(["aplay", "-q", "-D", args.device, "-t", "raw", "-f", "S16_LE",
                             "-r", str(RATE), "-c", "1"], stdin=subprocess.PIPE)
    try:
        while True:
            proc.stdin.write(pcm)
            proc.stdin.flush()
            if not args.loop:
                break
    except (KeyboardInterrupt, BrokenPipeError):
        pass
    finally:
        try:
            proc.stdin.close()
        except BrokenPipeError:
            pass
        proc.wait()


if __name__ == "__main__":
    main()
