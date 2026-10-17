"""Voice-controlled Voice Bonnet LED: say "turn on the light" or "turn off the light".

  voice_light.py                # both onboard LEDs, bonnet mic (plughw:1,0)
  voice_light.py --led 0        # only AIY_LED0
  voice_light.py --test         # blink the LEDs once and exit

Speech is recognised on the Pi with Vosk (offline), limited to the two phrases so nothing else
triggers it. The LEDs are the two on the bonnet board itself, lines AIY_LED0 / AIY_LED1 of the
bonnet's GPIO expander (gpio-aiy-io); driving them needs only membership of the gpio group.
(The button header's RGB LED is a separate KTD2026 chip and only lights with a button attached.)

Setup (no root; the venv needs the system's python3-libgpiod, hence --system-site-packages):
  python3 -m venv --system-site-packages ~/.local/share/voicelight/venv
  ~/.local/share/voicelight/venv/bin/pip install vosk
  cd ~/.local/share/voicelight && curl -fLO https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip
  python3 -m zipfile -e vosk-model-small-en-us-0.15.zip . && rm vosk-model-small-en-us-0.15.zip
"""
import argparse
import glob
import json
import subprocess
import sys
import time
from pathlib import Path

import gpiod
from gpiod.line import Direction, Value

RATE = 16000
PHRASES = {"turn on the light": True, "turn off the light": False}
DEFAULT_MODEL = Path.home() / ".local/share/voicelight/vosk-model-small-en-us-0.15"
MIC_ROUTE = ["Sto1 ADC MIXL ADC1", "Sto1 ADC MIXR ADC1", "RECMIXL BST1", "RECMIXR BST1",
             "RECMIXL BST2", "RECMIXR BST2"]


class BonnetLEDs:
    """The bonnet's onboard LEDs (active low: the line pulled low lights them), held as outputs
    for as long as this runs."""

    def __init__(self, names=("AIY_LED0", "AIY_LED1")):
        for path in sorted(glob.glob("/dev/gpiochip*")):
            with gpiod.Chip(path) as chip:
                if chip.get_info().label == "gpio-aiy-io":
                    self.offsets = [chip.line_offset_from_id(n) for n in names]
                    break
        else:
            sys.exit("Voice Bonnet GPIO expander (gpio-aiy-io) not found")
        self.request = gpiod.request_lines(
            path, consumer="voicelight",
            config={tuple(self.offsets): gpiod.LineSettings(direction=Direction.OUTPUT, active_low=True,
                                                            output_value=Value.INACTIVE)})

    def set(self, on):
        value = Value.ACTIVE if on else Value.INACTIVE
        self.request.set_values({o: value for o in self.offsets})


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--device", default="plughw:1,0", help="ALSA capture device (default: bonnet mic)")
    p.add_argument("--model", default=str(DEFAULT_MODEL), help="Vosk model folder")
    p.add_argument("--led", choices=["0", "1", "both"], default="both", help="which onboard LED(s)")
    p.add_argument("--test", action="store_true", help="blink the LED(s) once and exit")
    args = p.parse_args()

    leds = BonnetLEDs({"0": ("AIY_LED0",), "1": ("AIY_LED1",), "both": ("AIY_LED0", "AIY_LED1")}[args.led])
    if args.test:
        leds.set(True); time.sleep(1.5); leds.set(False)
        return

    from vosk import KaldiRecognizer, Model, SetLogLevel
    SetLogLevel(-1)
    if not Path(args.model).is_dir():
        sys.exit(f"Vosk model not found at {args.model} (see setup in this file's docstring)")
    rec = KaldiRecognizer(Model(args.model), RATE, json.dumps(list(PHRASES) + ["[unk]"]))

    card = args.device.split(":")[-1].split(",")[0]
    if card.isdigit() and "voicebonnet" in Path(f"/proc/asound/card{card}/id").read_text(errors="ignore"):
        for c in MIC_ROUTE:  # a reboot clears the bonnet's mic route
            subprocess.run(["amixer", "-c", card, "-q", "sset", c, "on"], stderr=subprocess.DEVNULL)

    mic = subprocess.Popen(["arecord", "-q", "-D", args.device, "-f", "S16_LE", "-r", str(RATE),
                            "-c", "1", "-t", "raw"], stdout=subprocess.PIPE)
    print('Listening: say "turn on the light" or "turn off the light" (Ctrl-C to stop)', flush=True)
    try:
        while True:
            data = mic.stdout.read(4000)  # 125 ms
            if not data:
                sys.exit("microphone stopped (is another program using it?)")
            if rec.AcceptWaveform(data):
                text = json.loads(rec.Result()).get("text", "")
                if text in PHRASES:
                    leds.set(PHRASES[text])
                    print(f"{time.strftime('%H:%M:%S')}  heard \"{text}\" -> light {'on' if PHRASES[text] else 'off'}",
                          flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        mic.terminate()


if __name__ == "__main__":
    main()
