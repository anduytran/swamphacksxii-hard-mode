#!/usr/bin/env bash
# Live demo: the real wearable client on the audio Pi (sreer), on the latest code, with its log on screen.
#
#     device/demo_live.sh [user@host]          default sreer@100.78.33.50
#
# Talk to the wearable ("Last Scene, I'm putting my pin on the desk", "Last Scene, where is my pin?",
# "Last Scene, this is meeting room 2360"): every request posts live to the API and shows up on the dashboard
# (Live, Timeline, Items, Map) within seconds.
#
# The client runs as the Pi's lastscene-device service, so it keeps listening when this window closes or the
# laptop disconnects, and restarts itself if it crashes. Ctrl-C here only stops showing the log.
#
# Steps on the Pi: pull the repo, copy the client, its speaker EQ and sounds into /opt/lastscene, install (or
# refresh) the service, restart it, check it's listening and the API is up, then follow its log.
set -euo pipefail

HOST="${1:-sreer@100.78.33.50}"
API="https://www.lastscene.photo"

echo "== API"
curl -s -m 10 "$API/healthz" | python3 -c '
import json, sys
d = json.load(sys.stdin)
speech = d.get("speech") or {}
print("   healthy:", d.get("ok"), "| positioning:", " then ".join(d.get("wifi_positioning_providers") or []) or "off",
      "| raia:", "on" if d.get("raia_context") else "off", "| voice:", speech.get("voice_model") or speech.get("provider"))
' || echo "   API not answering"

echo "== Wearable ($HOST): you'll be asked for its password, then sudo's"
ssh -t "$HOST" "set -e
cd ~/lastscene-src && git pull -q && echo \"   code: \$(git log --oneline -1 | cut -c1-90)\"
sudo cp -r device/lastscene_device.py device/speaker_eq.py device/sounds /opt/lastscene/
# The repo's unit runs as a lastscene user; this Pi runs the client as root (audio, GPIO, Wi-Fi scans).
sed -e 's/^User=lastscene/User=root/' -e '/^SupplementaryGroups=/d' device/lastscene-device.service \
  | sudo tee /etc/systemd/system/lastscene-device.service >/dev/null
sudo systemctl daemon-reload
sudo pkill -f '^/opt/lastscene/venv/bin/python /opt/lastscene/lastscene_device.py' || true  # a copy run by hand
sudo systemctl enable -q lastscene-device
sudo systemctl restart lastscene-device
echo '   starting (the wake model takes ~10 s)...'
for i in \$(seq 1 30); do
  sudo journalctl -u lastscene-device --since '-40 s' -o cat | grep -q 'LastScene ready' && break
  sleep 1
done
if sudo journalctl -u lastscene-device --since '-40 s' -o cat | grep -q 'LastScene ready'; then
  echo '   READY: say \"Last Scene\", wait for the chime, then speak.'
else
  echo '   NOT READY: the last lines of its log:'; sudo journalctl -u lastscene-device -n 15 --no-pager -o cat
fi
echo '== Live log (Ctrl-C stops the log, not the wearable)'
sudo journalctl -u lastscene-device -f -n 0 -o cat"
