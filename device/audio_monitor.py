"""Live audio monitor for the Voice Bonnet, in the terminal.

  python3 audio_monitor.py              # card 1 (aiy-voicebonnet)
  python3 audio_monitor.py --card 2

Shows the microphone level (RMS and peak, with a short history) and how much of each note of the
C major scale the mic hears, updated about 20 times a second. A slider sets the speaker volume.

Keys:  left/right  speaker volume     s  play the scale     t  tone on/off (440 Hz)
       up/down     mic boost          q  quit

On start it turns on the bonnet's speaker path and mic route (a reboot clears both); no root needed.
"""
import argparse
import array
import curses
import math
import re
import struct
import subprocess
import threading
import time

RATE = 16000
CHUNK = 800  # 50 ms
NOTES = [261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88, 523.25]
NAMES = ["C4", "D4", "E4", "F4", "G4", "A4", "B4", "C5"]
SPEAKER_PATH = ["DAC1 MIXL DAC1", "DAC1 MIXR DAC1", "Stereo DAC MIXL DAC L1", "Stereo DAC MIXR DAC R1",
                "SPK MIXL DAC L1", "SPK MIXR DAC R1", "SPKVOL L", "SPKVOL R",
                "SPOL MIX SPKVOL L", "SPOR MIX SPKVOL R", "Speaker Channel"]
MIC_ROUTE = ["Sto1 ADC MIXL ADC1", "Sto1 ADC MIXR ADC1", "RECMIXL BST1", "RECMIXR BST1",
             "RECMIXL BST2", "RECMIXR BST2"]


class Mixer:
    def __init__(self, card):
        self.card = str(card)

    def set(self, control, value):
        subprocess.run(["amixer", "-c", self.card, "-q", "sset", control, str(value)],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def get(self, control):
        """(value, max, dB text) of the first channel, or None if the control doesn't exist."""
        out = subprocess.run(["amixer", "-c", self.card, "sget", control],
                             capture_output=True, text=True).stdout
        # "Limits: Playback 0 - 39" / "Limits: 0 - 12"
        # "Front Left: Playback 31 [79%] [0.00dB] [on]" / "Mono: 0 [0%] [0.00dB]"
        top = re.search(r"Limits:.*?(\d+)\s*$", out, re.M)
        val = re.search(r"(?:Front Left|Mono):\s*(?:Playback\s+)?(\d+)\s*\[\d+%\](?:\s*\[(-?[\d.]+)dB\])?", out)
        if not top or not val:
            return None
        return int(val.group(1)), int(top.group(1)), f"{float(val.group(2)):+.1f} dB" if val.group(2) else ""


class Mic(threading.Thread):
    """Reads the mic in 50 ms chunks and keeps the latest levels."""

    def __init__(self, device):
        super().__init__(daemon=True)
        self.device = device
        self.rms_db = self.peak_db = -90.0
        self.notes = [-90.0] * len(NOTES)
        self.history = []
        self.error = ""
        self.lock = threading.Lock()

    def run(self):
        proc = subprocess.Popen(["arecord", "-q", "-D", self.device, "-f", "S16_LE", "-r", str(RATE),
                                 "-c", "1", "-t", "raw"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        while True:
            data = proc.stdout.read(CHUNK * 2)
            if len(data) < CHUNK * 2:
                self.error = "microphone stopped: " + proc.stderr.read().decode(errors="replace").strip()[:80]
                return
            x = array.array("h", data)
            rms = math.sqrt(sum(v * v for v in x) / len(x))
            peak = max(abs(v) for v in x)
            notes = [db(goertzel(x, f)) for f in NOTES]
            with self.lock:
                self.rms_db, self.peak_db, self.notes = db(rms), db(peak), notes
                self.history = (self.history + [self.rms_db])[-200:]


def goertzel(x, f):
    k = 2 * math.cos(2 * math.pi * f / RATE)
    s1 = s2 = 0.0
    for v in x:
        s1, s2 = v + k * s1 - s2, s1
    return math.sqrt(max(s1 * s1 + s2 * s2 - k * s1 * s2, 0.0)) * 2 / len(x)


def db(amplitude):
    return 20 * math.log10(max(amplitude, 1) / 32768)


class Player:
    """Scale and tone on the speaker, one at a time."""

    def __init__(self, device):
        self.device = device
        self.proc = None
        self.what = ""

    def _start(self, what, pcm, loop=False):
        self.stop()
        self.proc = subprocess.Popen(["aplay", "-q", "-D", self.device, "-t", "raw", "-f", "S16_LE",
                                      "-r", "24000", "-c", "1"], stdin=subprocess.PIPE,
                                     stderr=subprocess.DEVNULL)
        self.what = what

        def feed(proc=self.proc):
            try:
                while True:
                    proc.stdin.write(pcm)
                    proc.stdin.flush()
                    if not loop:
                        break
                proc.stdin.close()
            except (BrokenPipeError, ValueError, OSError):
                pass
        threading.Thread(target=feed, daemon=True).start()

    def scale(self):
        out = bytearray()
        n = int(24000 * 0.3)
        for f in NOTES:
            for i in range(n):
                env = min(1, i / 240, (n - i) / 1200)
                out += struct.pack("<h", int(9000 * env * math.sin(2 * math.pi * f * i / 24000)))
        self._start("scale", bytes(out))

    def tone(self):
        if self.what == "tone" and self.playing():
            return self.stop()
        out = b"".join(struct.pack("<h", int(9000 * math.sin(2 * math.pi * 440 * i / 24000)))
                       for i in range(24000))  # exactly 440 cycles, so it loops without a click
        self._start("tone", out, loop=True)

    def playing(self):
        return self.proc is not None and self.proc.poll() is None

    def stop(self):
        if self.playing():
            self.proc.terminate()
        self.what = ""


def bar(width, frac, ch="#"):
    n = max(0, min(width, int(round(frac * width))))
    return ch * n + "-" * (width - n)


def level_frac(level_db, floor=-70.0):
    return (level_db - floor) / -floor


def draw(scr, mic, mixer, player, vol, boost):
    scr.erase()
    h, w = scr.getmaxyx()
    bw = max(10, min(60, w - 24))
    put = lambda y, x, s, a=0: y < h and scr.addnstr(y, x, s, max(0, w - x - 1), a)
    with mic.lock:
        rms, peak, notes, hist = mic.rms_db, mic.peak_db, list(mic.notes), list(mic.history)

    put(0, 0, "Voice Bonnet audio monitor", curses.A_BOLD)
    put(1, 0, "left/right volume   up/down mic boost   s scale   t tone   q quit", curses.A_DIM)

    if vol:
        v, top, vdb = vol
        put(3, 0, f"Speaker  [{bar(bw, v / top, '=')}] {v:>2}/{top}  {vdb}")
    else:
        put(3, 0, "Speaker  (no 'Speaker' control on this card)")
    if boost:
        put(4, 0, f"Mic boost {boost[0]}/{boost[1]}")
    put(4, 30, f"Playing: {player.what if player.playing() else '-'}")

    hot = curses.color_pair(2) if peak > -3 else curses.color_pair(1)
    put(6, 0, f"Mic RMS  [{bar(bw, level_frac(rms))}] {rms:6.1f} dBFS", curses.color_pair(1))
    put(7, 0, f"Mic peak [{bar(bw, level_frac(peak))}] {peak:6.1f} dBFS" + ("  CLIPPING" if peak > -1 else ""), hot)

    put(9, 0, "Heard per note (dBFS)")
    for i, (name, lvl) in enumerate(zip(NAMES, notes)):
        put(10 + i, 0, f"  {name}  [{bar(bw, level_frac(lvl, -90))}] {lvl:6.1f}", curses.color_pair(3))

    put(19, 0, "Mic RMS, last 10 s")
    spark = " .:-=+*#%@"
    line = "".join(spark[max(0, min(9, int(level_frac(v) * 9)))] for v in hist[-(w - 2):])
    put(20, 0, line)
    if mic.error:
        put(22, 0, mic.error, curses.color_pair(2))
    scr.refresh()


def main(scr, args):
    curses.curs_set(0)
    curses.start_color()
    curses.use_default_colors()
    curses.init_pair(1, curses.COLOR_GREEN, -1)
    curses.init_pair(2, curses.COLOR_RED, -1)
    curses.init_pair(3, curses.COLOR_CYAN, -1)
    scr.nodelay(True)

    mixer = Mixer(args.card)
    device = f"plughw:{args.card},0"
    for c in SPEAKER_PATH + MIC_ROUTE:
        mixer.set(c, "on")
    mic = Mic(device)
    mic.start()
    player = Player(device)

    vol, boost = mixer.get("Speaker"), mixer.get("IN1 Boost")
    last_poll = 0.0
    try:
        while True:
            key = scr.getch()
            if key in (ord("q"), 27):
                break
            if vol and key in (curses.KEY_LEFT, curses.KEY_RIGHT):
                v = max(0, min(vol[1], vol[0] + (1 if key == curses.KEY_RIGHT else -1)))
                mixer.set("Speaker", v)
                vol = mixer.get("Speaker")
            elif boost and key in (curses.KEY_UP, curses.KEY_DOWN):
                b = max(0, min(boost[1], boost[0] + (1 if key == curses.KEY_UP else -1)))
                mixer.set("IN1 Boost", b)
                mixer.set("IN2 Boost", b)
                boost = mixer.get("IN1 Boost")
            elif key == ord("s"):
                player.scale()
            elif key == ord("t"):
                player.tone()
            if time.time() - last_poll > 1:  # pick up changes made elsewhere, e.g. alsamixer
                vol, boost, last_poll = mixer.get("Speaker"), mixer.get("IN1 Boost"), time.time()
            draw(scr, mic, mixer, player, vol, boost)
            time.sleep(0.05)
    finally:
        player.stop()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--card", type=int, default=1, help="ALSA card number of the Voice Bonnet (default 1)")
    try:
        curses.wrapper(main, p.parse_args())
    except KeyboardInterrupt:
        pass
