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
   Time it on the Pi before relying on it: `arecord -f S16_LE -r 16000 -c 1 -d 4 said.wav`, then `STT_ENGINE=moonshine /opt/lastscene/venv/bin/python lastscene_device.py --transcribe said.wav` (no network or Auth0 needed). On a Mac, one thread, Moonshine took 40–90 ms per 2 s clip with ~285 MB peak memory. If the model is missing or hears nothing, the clip goes to the server's speech-to-text instead; `STT_FALLBACK=false` turns that off.

   **Measured on a Pi Zero 2 W** (Oct 17, 2026; Debian 13, Python 3.13, onnxruntime 1.31.0, Moonshine tiny quantized, `LocalSTT`
   as the service uses it; clips are 16 kHz macOS voices, all transcribed word for word):

   | Clip | Alone | With the vision detector running (`lastscene_pipeline.bench`, 4 threads) |
   |---|---|---|
   | 1.2 s ("Where are my keys?") | 1.0 s | 2.7 s |
   | 3.5 s | 2.8 s | 4.8 s |
   | 6.8 s | 5.6 s | 10.3 s |
   | 10.8 s | 7.6 s | 14.7 s |
   | 14.7 s (about `MAX_RECORD_SECONDS`) | 10.1 s | 25.8 s |

   - Alone, Moonshine runs at 0.7–0.8x real time; the reply waits about as long as you spoke. A live 5 s recording from
     the USB camera's mic took 1.1–1.9 s.
   - Loading the model takes 9 s (17 s while the detector runs): once, when the service starts.
   - Memory: 168 MB after loading, peaks of 200–270 MB while transcribing (longer clips use more). The Pi has 415 MB
     usable, so it dropped to 35–44 MB available and used up to 66 MB of swap; nothing was killed.
   - Together, both slow down about 2–2.5x: the detector went from 0.63 s to 1.75 s per frame (p95 2.7 s), so keep
     detect at `--fps 1` or lower, and short questions answer much faster than 15 s monologues.

   **Households in another language** (set on the dashboard's Household page): Moonshine and Vosk as set up here are English-only, so set `STT_ENGINE=server` (the default) and the server's Gemini speech-to-text hears any language. Answers then come back spoken in the household's language. Saying "Last Scene" still wakes it in any language.
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

## Remote access (SSH from another network)

On the same Wi-Fi, `ssh <user>@<hostname>.local` works. From anywhere else it doesn't: a phone hotspot or home
router blocks incoming connections. [Tailscale](https://tailscale.com) (free for personal use) puts the Pi and
your laptop on one private network that works from any Wi-Fi, with no router setup.

1. **On the Pi** (while still on the same network as it; needs its sudo password once):
   ```bash
   curl -fsSL https://tailscale.com/install.sh | sh
   sudo tailscale up
   ```
   It prints a login link: open it and sign in. The Pi joins your tailnet and comes back after reboots.
2. **On the laptop**: install Tailscale (Mac: `brew install --cask tailscale-app`, or the App Store) and sign in
   to the same account from its menu-bar icon.
3. **Connect from anywhere** with MagicDNS's short name (the Pi's hostname), or its `100.x.y.z` address from
   `tailscale ip -4`. For a Pi named `faker` with the key `~/.ssh/faker`, a second SSH entry keeps the key
   working both ways:
   ```
   Host faker-remote
       HostName faker
       User faker
       IdentityFile ~/.ssh/faker
       HostKeyAlias faker.local
   ```
   `ssh faker` on the same Wi-Fi, `ssh faker-remote` elsewhere. `rsync`/`scp` work the same way.
   `HostKeyAlias` checks the Pi against the host key already saved for `faker.local` (it's the same Pi);
   without it the first connection fails with "Host key verification failed". Compare the two once with
   `ssh-keyscan -t ed25519 <100.x.y.z> | ssh-keygen -lf -` and `ssh-keygen -lF faker.local`.
   Tested Oct 17, 2026: from a different Wi-Fi than the Pi's phone hotspot, `ssh faker-remote` logged straight in.

- The Pi still needs internet: it only joins Wi-Fi networks it knows (Raspberry Pi Imager's, plus any added with
  `sudo nmcli device wifi connect <SSID> password <password>`).
- Browser-only alternative: Raspberry Pi Connect (`sudo apt install rpi-connect-lite`, `rpi-connect on`,
  `rpi-connect signin`) gives a shell at connect.raspberrypi.com, but no `ssh`, `scp` or `rsync` from the laptop.

## Item detection on the device

With `VISION_DIR` set, the service also runs the vision pipeline's detector (`vision/lastscene_pipeline`, the
INT8 model from `train`/`fetch` under `LASTSCENE_DATA/models/current`), sharing one camera and one detector:

1. **You speak.** Wake phrase or REMEMBER button, then the words are transcribed on the Pi.
2. **A question** ("Where are my keys?") is sent with no photo. The device uses the server's own rules
   (`backend/app/intent.py`, mirrored and tested in `test_device.py`), because the server rejects a statement
   without a photo.
3. **A statement** ("I'm putting my pin on the desk") gets its photo *after* the words. If it names a taught item
   (class name, plural, or Teach alias) and the photo doesn't show it at `MIN_CONFIDENCE`, the camera tries
   `CAPTURE_RETRIES` (3) more times `RETRY_GAP_SECONDS` (1 s) apart, with a short low beep on the first retry
   ("point me at it"), and sends the first photo that shows it, or the last one. No taught item named: one photo.
4. **Between conversations**, a frame every `VISION_EVERY` seconds (2) goes to the detector only if the scene
   changed (`lastscene_pipeline/gated.py`), plus 6 s after a change so a just-placed item is seen at rest, and
   once a minute regardless. Sightings are posted as observations, as `python -m lastscene_pipeline.detect` does.
   It pauses while you talk to the device.

Try it without uploading anything (camera + model only):
```bash
export VISION_DIR=~/swamphacksxii-hard-mode/vision LASTSCENE_DATA=~/lastscene-data
python3 lastscene_device.py --capture "I'm putting my pin on the desk"   # retries, saves capture.jpg
python3 lastscene_device.py --watch 120                                  # background loop, observations printed
```

**Measured on a Pi Zero 2 W** (Oct 17, 2026; USB camera, 1-class Pin model, YOLO11s INT8 at 320 px):

| | |
|---|---|
| Statement photo, item not in view (1 + 3 retries) | 6.0–6.5 s, then the last photo is sent |
| One attempt (grab + detect) | ~0.75 s |
| Questions | no photo, no detector run |
| Detector, alone / with speech-to-text in the same process | 0.64 s / 1.13 s a frame (paused during conversations: 0.69 s) |
| Speech-to-text, 3.5 s clip, detector paused vs running | 2.8 s vs 4.5–5.4 s |
| Peak memory, one process (speech-to-text + detector) | 250–290 MB (two processes: ~355 MB) |
| Frame change every 2 s: still view / busy room | 0.3–1.4 / 2.4–44 (median 9.7); the gate threshold is 4 |
| Background loop, 2 min facing a busy room | 56 of 60 frames checked, ~0.9 CPU cores, 50 °C |

- **USB camera:** the frame waiting in the driver's buffer is discarded before each photo. With the detector
  running between reads, the Zero 2 W's USB controller drops camera data: 19 of 25 MJPEG 1280x720 frames came out
  corrupt (torn bands; uncompressed frames tear too, silently). Reading the next frame instead: 0 of 25, ~90 ms.
- **ONNX Runtime** runs without spin-waiting on boards under 1 GB (`runtime.load_detector`'s low-memory mode):
  4% slower alone (0.64 vs 0.615 s), but 21% less CPU, which is what speech-to-text needs. FP32 was slower
  (0.72 s), 3 threads slower (0.74 s), and the pip wheel has no XNNPACK provider.
- **Memory is the limit:** free memory dipped to 16–67 MB during a 15 s clip with both models loaded. Keep
  `MAX_RECORD_SECONDS` short, or set `VISION_DIR=` empty on a board that also runs other things.

## Two Pis: camera on one, audio on the other

The wearable can be split across two Pi Zero 2 Ws: **faker** has the camera (and runs the item detector),
**sreer** has the mic, speaker, buttons and LEDs and runs the client. They talk as little as possible: for
each memory sreer makes **one** request and faker answers with one JPEG (~100–200 KB). Questions never
involve faker, and nothing is streamed or polled in between.

- **faker (camera):** copy `lastscene_device.py` and `camera_server.py` (and the `vision/` folder for item
  detection) to `/opt/lastscene`. In `/etc/lastscene/camera.env` set `CAMERA_TOKEN` (make one with
  `python3 -c "import secrets; print(secrets.token_urlsafe(24))"`), `CAMERA` / `CAMERA_INDEX`, and for item
  detection `VISION_DIR` and `LASTSCENE_DATA` (plus `API_URL` and the device's Auth0 settings if its background
  detection should upload). Then:
  ```bash
  sudo cp lastscene-camera.service /etc/systemd/system/ && sudo systemctl enable --now lastscene-camera
  ```
  `GET /photo?for=<words>` retries until the item the words name is in the photo; `GET /health` reports
  the camera and model. Every request needs the header `X-Camera-Token`.
- **sreer (audio):** the normal setup above, with no camera and no `VISION_DIR`. In `/etc/lastscene/device.env`:
  `CAMERA=remote`, `CAMERA_URL=http://faker:8765` (Tailscale MagicDNS; or `http://faker.local:8765` on the
  same Wi-Fi, or faker's `100.x` address), and the same `CAMERA_TOKEN`.
- **Check the link from sreer:** `curl -s -H "X-Camera-Token: <token>" http://faker:8765/health`.
- If faker can't be reached, a memory isn't saved and sreer plays the low beep; questions still work.

## Pairing and location

- **Pairing code.** Generate one per device and put it on the case label with the Client ID; set it as `PAIRING_CODE` in `/etc/lastscene/device.env`. The device sends it with every request; a caregiver enters the Client ID + code on the dashboard's **Household** page to add the wearable to their household.
- **Location (Wi-Fi positioning).** The Zero 2 W has no GPS. With `LOCATION_SOURCE=wifi` (default) it scans nearby access points (`nmcli`, falling back to `iw`, for which the service has `CAP_NET_ADMIN`) and sends BSSIDs + signal strengths; the server turns that into a position with Google's Geolocation API and keeps only the coordinates. Check what it would send with `python3 lastscene_device.py --where`. `LOCATION_SOURCE=fixed` with `HOME_LAT`/`HOME_LNG` tags every memory with one place instead; `none` sends nothing.
- Memories spooled while offline keep the Wi-Fi scan from when they happened.

## Using it

- **Say "Last Scene"**, wait for the beep, then speak: "I'm putting my keys on the counter" is saved with a photo (high beep), "Where are my keys?" is answered out loud. It records until you stop talking (up to `WAKE_MAX_SECONDS`). Audio stays on the Pi until the phrase is heard; only what's said after the beep is sent (as text, with `STT_ENGINE` set). If it triggers by mistake, raise `WAKE_THRESHOLD`; if it misses you, lower it. `WAKE_WORD` points at the "Last Scene" model (training: [wake/README.md](wake/README.md)); `WAKE_WORD=hey_jarvis` works right away for testing, with "Hey Jarvis" as the phrase.
- **Live conversation** (`LIVE=true` here and `LIVE_ENABLED=true` on the server): say "Last Scene" (or hold ASK for `LIVE_HOLD_SECONDS` without speaking, then release), wait for the chime, and just talk: "Where are my keys?", "And my glasses?", "I'm putting my wallet in the drawer". Gemini looks things up in your memories (it's told never to guess) and saves new ones with a photo. It ends with a low beep after `LIVE_SILENCE_SECONDS` of quiet, at `LIVE_MAX_SECONDS`, or when you press any button. The mic is muted while Gemini speaks (the Pi has no echo cancellation), so wait for it to finish. Nothing is streamed before the wake phrase or the long press. If live mode can't start (server has it off, quota, no Wi-Fi), the normal one-shot flow runs instead. Check it with `--live-check`, which gets a token and opens a session without sending audio.
- **Hold REMEMBER**, say where you're putting something ("I'm putting my keys on the counter"), release. It takes a photo when you press (with item detection on: after the words, retried until the item you named is in it), records while held, and uploads when you let go. The LED is solid while recording and blinks while sending.
- **Hold ASK**, ask ("Where are my keys?"), release. The answer plays through the speaker.
- With no Wi-Fi, memories are kept in `/var/lib/lastscene/spool` and uploaded automatically once it's back. Questions need a connection.

## Notes

- The API must be reachable from the Pi over **HTTPS** with a real hostname. `localhost` on your laptop is not reachable; for testing on the same network use `http://<laptop-ip>:8000` and start uvicorn with `--host 0.0.0.0`.
- Unlike the ESP32 in the original plan, the Pi can't deep-sleep; budget for a battery that lasts the day, or accept a ~25 s boot if you power it on demand.
- The wake phrase and the buttons work side by side; buttons stay the more reliable way in for the target user.
- With `STT_ENGINE` set, a low beep after speaking means nothing was heard (and, with `STT_FALLBACK=false`, nothing was sent).
