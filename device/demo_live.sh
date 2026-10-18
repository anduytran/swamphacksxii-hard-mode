#!/usr/bin/env bash
# Live demo: both Pis on the latest code, the camera's live view in a browser, and the wearable's log on screen.
#
#     device/demo_live.sh [user@audio-pi] [user@camera-pi]   defaults sreer@100.78.33.50, faker@100.89.232.47
#     SKIP_CAMERA=1 device/demo_live.sh                       the wearable only
#
# Talk to the wearable ("Last Scene, I'm putting my pin on the desk", "Last Scene, where is my pin?",
# "Last Scene, this is meeting room 2360"): every request posts live to the API and shows up on the dashboard
# (Live, Timeline, Items, Map) within seconds, and the camera view shows what passive detection sees.
#
# 1. Camera Pi (faker): pull the repo, restart its camera service, check it's detecting, and start the camera
#    view on this laptop (http://127.0.0.1:8766, opened in the browser; it keeps running after this script:
#    stop it with  pkill -f camera_view.py ).
# 2. Audio Pi (sreer): pull the repo, copy the client, its speaker EQ and sounds into /opt/lastscene, install
#    (or refresh) the lastscene-device service, restart it, check it's listening, then follow its log.
# Both clients run as services, so they keep working when this window closes; Ctrl-C only stops the log.
# Each Pi asks for its password once, then sudo's.
set -euo pipefail

HOST="${1:-sreer@100.78.33.50}"
CAMERA_HOST="${2:-faker@100.89.232.47}"
API="https://www.lastscene.photo"
VIEW_PORT=8766
HERE="$(cd "$(dirname "$0")" && pwd)"
SOCKETS="$(mktemp -d /tmp/lsd.XXXX)"  # short: macOS caps a socket path at 104 characters
trap 'rm -rf "$SOCKETS"' EXIT
ssh_to() {  # one login per Pi for the whole run
  ssh -o ControlMaster=auto -o ControlPath="$SOCKETS/%C" -o ControlPersist=120 "$@"
}

echo "== API"
curl -s -m 10 "$API/healthz" | python3 -c 'import json, sys; print("   healthy:", json.load(sys.stdin).get("ok"))' \
  || echo "   API not answering"

if [ -z "${SKIP_CAMERA:-}" ]; then
  echo "== Camera Pi ($CAMERA_HOST): its password, then sudo's"
  CAMERA_IP="${CAMERA_HOST#*@}"
  # The token is root-only on the Pi: copied to a private temp file there, read over the same login, deleted.
  ssh_to -t "$CAMERA_HOST" "set -e
cd ~/lastscene-src && git pull -q && echo \"   code: \$(git log --oneline -1 | cut -c1-90)\"
sudo systemctl restart lastscene-camera
umask 077 && sudo grep '^CAMERA_TOKEN=' /etc/lastscene/camera.env | cut -d= -f2- > /tmp/.lastscene-camtok"
  TOKEN="$(ssh_to "$CAMERA_HOST" 'cat /tmp/.lastscene-camtok; rm -f /tmp/.lastscene-camtok')"
  echo "   starting (the detector takes ~20 s)..."
  HEALTH=""
  for _ in $(seq 1 40); do
    HEALTH="$(curl -s -m 3 -H "X-Camera-Token: $TOKEN" "http://$CAMERA_IP:8765/health" || true)"
    [ -n "$HEALTH" ] && break
    sleep 1
  done
  echo "$HEALTH" | python3 -c '
import json, sys
try:
    h = json.load(sys.stdin)
except ValueError:
    print("   NOT READY: the camera Pi did not answer"); sys.exit()
p = h.get("passive") or {}
print("   camera:", h.get("camera"), "| detector:", h.get("vision") or "none", "| passive detection:", p.get("state"),
      ("(" + p["error"] + ")") if p.get("error") else "")'
  pkill -f "camera_view.py" 2>/dev/null || true  # an older view, holding the port
  sleep 1
  CAMERA_TOKEN="$TOKEN" nohup python3 "$HERE/camera_view.py" --camera "http://$CAMERA_IP:8765" --port "$VIEW_PORT" \
    >"${TMPDIR:-/tmp}/lastscene-camera-view.log" 2>&1 &
  sleep 1
  echo "   camera view: http://127.0.0.1:$VIEW_PORT  (keeps running; stop with: pkill -f camera_view.py)"
  command -v open >/dev/null && open "http://127.0.0.1:$VIEW_PORT" || true
fi

echo "== Wearable ($HOST): its password, then sudo's"
ssh_to -t "$HOST" "set -e
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
