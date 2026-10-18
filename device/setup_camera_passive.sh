#!/usr/bin/env bash
# Turn on passive item detection on the camera Pi (faker): its background loop posts each confirmed sighting as an
# observation, and the server updates the item's memory and reads the frame with raia, silently.
#
# On faker, as root (prints key names only, never values):
#     sudo bash setup_camera_passive.sh [git ref] [credentials file]
#
#   git ref           the repo commit/branch to install device/lastscene_device.py + camera_server.py from (master)
#   credentials file  KEY=VALUE lines merged into /etc/lastscene/camera.env, then deleted (/tmp/lastscene-passive.env):
#                     API_URL, AUTH0_DOMAIN, AUTH0_AUDIENCE, DEVICE_CLIENT_ID, DEVICE_CLIENT_SECRET, PAIRING_CODE
#
# Then restarts lastscene-camera and shows what it logged about passive detection and its /health.
set -euo pipefail

REF="${1:-master}"
CREDS="${2:-/tmp/lastscene-passive.env}"
REPO="https://raw.githubusercontent.com/anduytran/swamphacksxii-hard-mode"
APP="${APP:-/opt/lastscene}"
ENVF="${ENVF:-/etc/lastscene/camera.env}"
STATE="${STATE:-/var/lib/lastscene}"
SERVICE=lastscene-camera

[ "$(id -u)" -eq 0 ] || [ -n "${SETUP_TEST:-}" ] || exec sudo bash "$0" "$@"
PY="$APP/venv/bin/python"
[ -x "$PY" ] || PY=python3

echo "== 1. Device code from $REF"
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
for f in lastscene_device.py camera_server.py; do
  curl -fsSL "$REPO/$REF/device/$f" -o "$tmp/$f"
done
"$PY" -m py_compile "$tmp/lastscene_device.py" "$tmp/camera_server.py"
mkdir -p "$APP"
stamp=$(date +%Y%m%d-%H%M%S)
for f in lastscene_device.py camera_server.py; do
  [ -f "$APP/$f" ] && cp -p "$APP/$f" "$APP/$f.bak-$stamp"
  install -m 0644 "$tmp/$f" "$APP/$f"
done
echo "   installed in $APP (old copies: *.bak-$stamp)"

echo "== 2. Device credentials into $ENVF"
mkdir -p "$(dirname "$ENVF")" "$STATE"
touch "$ENVF"
cp -p "$ENVF" "$ENVF.bak-$stamp"
set_var() {  # KEY VALUE: replace the KEY= line, or add it
  local key="$1" value="$2" out
  out=$(mktemp)
  grep -v "^${key}=" "$ENVF" > "$out" || true
  printf '%s=%s\n' "$key" "$value" >> "$out"
  cat "$out" > "$ENVF"
  rm -f "$out"
}
if [ -f "$CREDS" ]; then
  while IFS= read -r line || [ -n "$line" ]; do
    case "$line" in ''|\#*) continue ;; esac
    key="${line%%=*}"
    case "$key" in
      API_URL|AUTH0_DOMAIN|AUTH0_AUDIENCE|DEVICE_CLIENT_ID|DEVICE_CLIENT_SECRET|PAIRING_CODE)
        set_var "$key" "${line#*=}"; echo "   set $key" ;;
      *) echo "   skipped $key (not a passive-detection setting)" ;;
    esac
  done < "$CREDS"
  shred -u "$CREDS" 2>/dev/null || rm -f "$CREDS"
else
  echo "   no $CREDS: keeping what $ENVF has"
fi
# The Auth0 token is cached here; the service user may have no home folder.
set_var LASTSCENE_TOKEN_CACHE "$STATE/auth0.json"
if id lastscene >/dev/null 2>&1; then chown lastscene "$STATE"; fi
chmod 600 "$ENVF"
echo "   $ENVF now has: $(cut -d= -f1 "$ENVF" | grep -v '^#' | grep . | tr '\n' ' ')"
grep -q '^VISION_DIR=.' "$ENVF" || echo "   WARNING: VISION_DIR is empty: no item detector, so no passive detection"
for key in LASTSCENE_DATA VISION_EVERY MIN_CONFIDENCE; do
  grep -q "^${key}=" "$ENVF" || echo "   note: $key not set, its default is used"
done

echo "== 3. Restart $SERVICE"
if systemctl list-unit-files "$SERVICE.service" >/dev/null 2>&1 && systemctl cat "$SERVICE" >/dev/null 2>&1; then
  since=$(date '+%Y-%m-%d %H:%M:%S')
  systemctl restart "$SERVICE"
  echo "   waiting for the detector to load..."
  for _ in $(seq 1 30); do
    journalctl -u "$SERVICE" --since "$since" --no-pager 2>/dev/null | grep -qE "Camera ready|Passive item detection" && break
    sleep 2
  done
  sleep 3
  journalctl -u "$SERVICE" --since "$since" --no-pager -o cat 2>/dev/null \
    | grep -E "Passive item detection|Item detection|Camera ready|detector:|Background item detection|posting|Traceback|Error" \
    | tail -20 | sed 's/^/   /'
else
  echo "   no $SERVICE systemd unit here: start camera_server.py the way you usually do, with $ENVF loaded"
fi

echo "== 4. Camera /health"
token=$(grep '^CAMERA_TOKEN=' "$ENVF" | tail -1 | cut -d= -f2- || true)
port=$(grep '^CAMERA_PORT=' "$ENVF" | tail -1 | cut -d= -f2- || true)
curl -s -m 10 -H "X-Camera-Token: $token" "http://127.0.0.1:${port:-8765}/health" | sed 's/^/   /' \
  || echo "   no answer on port ${port:-8765} (is camera_server.py running?)"
echo
echo "   \"passive\": {\"state\": \"running\"} means sightings are being posted; anything else says why not."
