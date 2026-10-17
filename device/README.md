# LastScene wearable — Raspberry Pi Zero 2 W

The Pi is a thin client: it captures a photo and voice, sends them to the API, and plays back the spoken answer. All the ML (speech-to-text, embeddings, search, Claude, text-to-speech) runs in the cloud, so the Zero 2 W's 512 MB of RAM is plenty.

```
 [REMEMBER button]──┐                         ┌── POST /api/v1/memory/log   (WAV + JPEG)
 [ASK button]───────┤   Pi Zero 2 W           │
 [I2S mic]──────────┼── lastscene_device.py ──┼── POST /api/v1/memory/query (WAV) → WAV answer
 [Camera]───────────┤   (Auth0 M2M token)     │
 [I2S amp+speaker]──┘                         └── https://<your API>
```

## Parts

| Part | Example | Notes |
|---|---|---|
| Raspberry Pi Zero 2 W | with soldered headers | |
| Camera | Camera Module 3 (or v2) | needs the **Zero** ribbon cable (22-pin → 15-pin) |
| Microphone | INMP441 I2S MEMS mic | or any USB mic via a micro-USB OTG adapter |
| Speaker amp | MAX98357A I2S amp + 4–8 Ω speaker | or a USB audio dongle |
| 2 × momentary buttons | 6 mm tactile or arcade | REMEMBER and ASK |
| LED + 330 Ω resistor | | recording / sending indicator |
| Power | 5 V 2.5 A supply, or a PiSugar / LiPo + 5 V boost | a Zero 2 W draws ~0.6–1 W idle |

> The USB route (USB mic + USB speaker through a hub/OTG) skips all audio wiring and is the fastest way to a working demo. I2S is smaller and better for a wearable.

## Wiring (BCM numbering, physical pin in brackets)

**Buttons and LED** (buttons use the Pi's internal pull-ups, so each just shorts the pin to ground):

| Signal | Pi pin |
|---|---|
| REMEMBER button | GPIO17 [11] ↔ GND [9] |
| ASK button | GPIO27 [13] ↔ GND [14] |
| LED (+ 330 Ω) | GPIO22 [15] → LED → GND [20] |

**I2S audio** (mic and amp share the clock lines):

| Signal | Pi pin | INMP441 mic | MAX98357A amp |
|---|---|---|---|
| 3.3 V | [1] | VDD | — |
| 5 V | [2] | — | Vin |
| GND | [6] | GND, L/R (→ left channel) | GND |
| Bit clock | GPIO18 [12] | SCK | BCLK |
| Word select | GPIO19 [35] | WS | LRC |
| Data in (mic → Pi) | GPIO20 [38] | SD | — |
| Data out (Pi → amp) | GPIO21 [40] | — | DIN |

**Camera:** ribbon into the Zero's CSI connector, contacts facing the board.

## Set up the Pi

1. Flash **Raspberry Pi OS Lite (64-bit)** with Raspberry Pi Imager. In its settings, set the Wi-Fi network, hostname (`lastscene`), and enable SSH.
2. Enable I2S audio. Add to `/boot/firmware/config.txt`, then reboot:
   ```
   dtparam=i2s=on
   dtoverlay=googlevoicehat-soundcard
   ```
   This overlay drives an I2S mic on GPIO20 and an I2S amp on GPIO21 at the same time. Check with `arecord -l` and `aplay -l`. (Skip this step if you're using USB audio.)
3. Install system packages, and a small virtual environment for the wake phrase (openWakeWord isn't in apt):
   ```bash
   sudo apt update && sudo apt install -y python3-picamera2 python3-gpiozero python3-httpx python3-numpy python3-venv alsa-utils
   sudo python3 -m venv --system-site-packages /opt/lastscene/venv
   sudo /opt/lastscene/venv/bin/pip install openwakeword
   sudo /opt/lastscene/venv/bin/python -c "from openwakeword.utils import download_models; download_models(['hey_jarvis'])"
   ```
   (That fetches openWakeWord's shared feature models; the "Last Scene" model itself ships in `wake/`.)
   The service runs from this venv. With `WAKE_WORD` empty, or its model missing, only the buttons work.

   **Speech-to-text on the Pi** (optional, recommended): with `STT_ENGINE=moonshine` the Pi transcribes what you said and sends only the words, so the recording never leaves the wearable. Moonshine tiny (quantized ONNX, 28 MB) is the default. `--no-deps` skips librosa/numba, which it only needs for reading audio files:
   ```bash
   sudo /opt/lastscene/venv/bin/pip install onnxruntime tokenizers
   sudo /opt/lastscene/venv/bin/pip install --no-deps useful-moonshine-onnx
   sudo mkdir -p /opt/lastscene/stt/moonshine-tiny && cd /opt/lastscene/stt/moonshine-tiny
   for f in encoder_model decoder_model_merged; do
     sudo curl -fL -o $f.onnx https://huggingface.co/UsefulSensors/moonshine/resolve/main/onnx/merged/tiny/quantized/$f.onnx
   done
   ```
   Or Vosk instead (`STT_ENGINE=vosk`; ~68 MB, faster to load, but lower case with no punctuation and less accurate):
   ```bash
   sudo /opt/lastscene/venv/bin/pip install vosk
   cd /opt/lastscene/stt && sudo curl -fLO https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip && sudo unzip vosk-model-small-en-us-0.15.zip
   ```
   **Live conversation** (optional, `LIVE=true`): talk back and forth with Gemini Live. The Pi connects to Gemini itself with a one-use token from the server (the server's Gemini key never reaches the Pi), so it needs the SDK:
   ```bash
   sudo /opt/lastscene/venv/bin/pip install google-genai
   ```
   Time it on the Pi before relying on it: `arecord -f S16_LE -r 16000 -c 1 -d 4 said.wav`, then `STT_ENGINE=moonshine /opt/lastscene/venv/bin/python lastscene_device.py --transcribe said.wav` (no network or Auth0 needed). On a Mac, one thread, Moonshine took 40–90 ms per 2 s clip with ~285 MB peak memory; the Zero 2 W has 512 MB in all, so check `free -m` with the service running. If the model is missing or hears nothing, the clip goes to the server's speech-to-text instead; `STT_FALLBACK=false` turns that off.
4. Copy this folder to the Pi and set up config:
   ```bash
   sudo useradd --system --groups gpio,video,audio lastscene
   sudo mkdir -p /opt/lastscene /etc/lastscene /var/lib/lastscene/spool
   sudo chown lastscene /var/lib/lastscene/spool
   sudo cp -r lastscene_device.py wake /opt/lastscene/
   sudo install -m 640 -g lastscene device.env.example /etc/lastscene/device.env
   ```
   Fill in `/etc/lastscene/device.env` with the **Wearable** M2M app's Client ID/Secret and your API's HTTPS URL.
5. Test without buttons:
   ```bash
   set -a && . /etc/lastscene/device.env && set +a && python3 /opt/lastscene/lastscene_device.py --check
   ```
   Then record a test: `arecord -D default -f S16_LE -r 16000 -c 1 -d 4 test.wav`, take a photo with `rpicam-still -o test.jpg`, and run `--log-file test.wav test.jpg`.
6. Run it at boot:
   ```bash
   sudo cp lastscene-device.service /etc/systemd/system/ && sudo systemctl enable --now lastscene-device
   ```
   Logs: `journalctl -u lastscene-device -f`.

## Pairing and location

- **Pairing code.** Generate one per device and put it on the case label with the Client ID; set it as `PAIRING_CODE` in `/etc/lastscene/device.env`. The device sends it with every request; a caregiver enters the Client ID + code on the dashboard's **Household** page to add the wearable to their household.
- **Location (Wi-Fi positioning).** The Zero 2 W has no GPS. With `LOCATION_SOURCE=wifi` (default) it scans nearby access points (`nmcli`, falling back to `iw`, for which the service has `CAP_NET_ADMIN`) and sends BSSIDs + signal strengths; the server turns that into a position with Google's Geolocation API and keeps only the coordinates. Check what it would send with `python3 lastscene_device.py --where`. `LOCATION_SOURCE=fixed` with `HOME_LAT`/`HOME_LNG` tags every memory with one place instead; `none` sends nothing.
- Memories spooled while offline keep the Wi-Fi scan from when they happened.

## Using it

- **Say "Last Scene"**, wait for the beep, then speak: "I'm putting my keys on the counter" is saved with a photo (high beep), "Where are my keys?" is answered out loud. It records until you stop talking (up to `WAKE_MAX_SECONDS`). Audio stays on the Pi until the phrase is heard; only what's said after the beep is sent (as text, with `STT_ENGINE` set). If it triggers by mistake, raise `WAKE_THRESHOLD`; if it misses you, lower it. `WAKE_WORD` points at the "Last Scene" model (training: [wake/README.md](wake/README.md)); `WAKE_WORD=hey_jarvis` works right away for testing, with "Hey Jarvis" as the phrase.
- **Live conversation** (`LIVE=true` here and `LIVE_ENABLED=true` on the server): say "Last Scene" (or hold ASK for `LIVE_HOLD_SECONDS` without speaking, then release), wait for the chime, and just talk: "Where are my keys?", "And my glasses?", "I'm putting my wallet in the drawer". Gemini looks things up in your memories (it's told never to guess) and saves new ones with a photo. It ends with a low beep after `LIVE_SILENCE_SECONDS` of quiet, at `LIVE_MAX_SECONDS`, or when you press any button. The mic is muted while Gemini speaks (the Pi has no echo cancellation), so wait for it to finish. Nothing is streamed before the wake phrase or the long press. If live mode can't start (server has it off, quota, no Wi-Fi), the normal one-shot flow runs instead. Check it with `--live-check`, which gets a token and opens a session without sending audio.
- **Hold REMEMBER**, say where you're putting something ("I'm putting my keys on the counter"), release. It takes a photo when you press, records while held, and uploads when you let go. The LED is solid while recording and blinks while sending.
- **Hold ASK**, ask ("Where are my keys?"), release. The answer plays through the speaker.
- With no Wi-Fi, memories are kept in `/var/lib/lastscene/spool` and uploaded automatically once it's back. Questions need a connection.

## Notes

- The API must be reachable from the Pi over **HTTPS** with a real hostname. `localhost` on your laptop is not reachable; for testing on the same network use `http://<laptop-ip>:8000` and start uvicorn with `--host 0.0.0.0`.
- Unlike the ESP32 in the original plan, the Pi can't deep-sleep; budget for a battery that lasts the day, or accept a ~25 s boot if you power it on demand.
- The wake phrase and the buttons work side by side; buttons stay the more reliable way in for the target user.
- With `STT_ENGINE` set, a low beep after speaking means nothing was heard (and, with `STT_FALLBACK=false`, nothing was sent).
