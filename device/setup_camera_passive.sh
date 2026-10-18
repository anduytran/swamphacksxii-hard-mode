#!/usr/bin/env bash
# Turn on passive item detection on the camera Pi (faker): its background loop posts each confirmed sighting as an
# observation, and the server updates the item's memory and reads the frame with raia, silently.
#
# On faker (asks for sudo; prints key names only, never values):
#     bash setup_camera_passive.sh [credentials file]
#
#   credentials file  KEY=VALUE lines merged into /etc/lastscene/camera.env, then deleted (default
#                     /tmp/lastscene-passive.env; skipped if missing): API_URL, AUTH0_DOMAIN, AUTH0_AUDIENCE,
#                     DEVICE_CLIENT_ID, DEVICE_CLIENT_SECRET, PAIRING_CODE
#
# Works from the lastscene-camera unit as installed: its User, the checkout its camera_server.py runs from, and its
# Python. Updates that checkout (git pull), points VISION_DIR at its vision/ folder and LASTSCENE_DATA at the user's
# lastscene-data, downloads the household's current model, restarts the service, and shows its log and /health.
set -euo pipefail

CREDS="${1:-/tmp/lastscene-passive.env}"
ENVF="${ENVF:-/etc/lastscene/camera.env}"
STATE="${STATE:-/var/lib/lastscene}"
SERVICE=lastscene-camera

[ "$(id -u)" -eq 0 ] || exec sudo bash "$0" "$@"

prop() { systemctl show -p "$1" --value "$SERVICE"; }
RUN_AS=$(prop User); RUN_AS=${RUN_AS:-root}
EXEC=$(systemctl cat "$SERVICE" | sed -n 's/^ExecStart=//p' | tail -1)
PY=$(echo "$EXEC" | awk '{print $1}')
SERVER=$(echo "$EXEC" | awk '{print $2}')
SRC=$(cd "$(dirname "$SERVER")/.." && pwd)
HOME_OF=$(getent passwd "$RUN_AS" | cut -d: -f6)
as_user() { sudo -u "$RUN_AS" -H "$@"; }
echo "service $SERVICE: user $RUN_AS, python $PY, checkout $SRC"

echo "== 1. Update the checkout"
if as_user git -C "$SRC" rev-parse --git-dir >/dev/null 2>&1; then
  as_user git -C "$SRC" pull -q --ff-only
  echo "   $(as_user git -C "$SRC" log --oneline -1)"
else
  echo "   $SRC is not a git checkout: leaving it as is"
fi
VISION="$SRC/vision"
[ -d "$VISION/lastscene_pipeline" ] || { echo "   no $VISION/lastscene_pipeline: can't run the detector" >&2; exit 1; }
"$PY" -c "import onnxruntime, cv2, numpy" 2>/dev/null \
  || { echo "   $PY can't import onnxruntime/cv2/numpy: run vision/pi/install.sh first" >&2; exit 1; }
echo "   $PY has onnxruntime $("$PY" -c 'import onnxruntime; print(onnxruntime.__version__)')"

echo "== 2. Settings in $ENVF"
mkdir -p "$(dirname "$ENVF")" "$STATE"
touch "$ENVF"
stamp=$(date +%Y%m%d-%H%M%S)
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
fi
DATA="${HOME_OF}/lastscene-data"
set_var VISION_DIR "$VISION"
set_var LASTSCENE_DATA "$DATA"
set_var LASTSCENE_TOKEN_CACHE "$STATE/auth0.json"  # the Auth0 token, cached between restarts
chown "$RUN_AS" "$STATE"
chmod 600 "$ENVF"
echo "   VISION_DIR=$VISION, LASTSCENE_DATA=$DATA"
echo "   $ENVF now has: $(cut -d= -f1 "$ENVF" | grep -v '^#' | grep . | tr '\n' ' ')"
missing=""
for key in API_URL AUTH0_DOMAIN AUTH0_AUDIENCE DEVICE_CLIENT_ID DEVICE_CLIENT_SECRET; do
  grep -q "^${key}=." "$ENVF" || missing="$missing $key"
done
[ -z "$missing" ] || echo "   WARNING: missing$missing: sightings can't be posted"
if grep -q '^VISION_EVERY=0' "$ENVF"; then echo "   WARNING: VISION_EVERY=0 turns the background loop off"; fi

echo "== 3. The household's current model"
as_user mkdir -p "$DATA/models"
# The settings file is root-only, so it's handed to the fetch on stdin (already open, so no permission check).
as_user bash -c 'set -a; eval "$(cat)"; set +a; cd "$0" && exec "$1" -m lastscene_pipeline.fetch' \
  "$VISION" "$PY" < "$ENVF" 2>&1 | sed 's/^/   /' || true
echo "   current: $(cat "$DATA/models/current" 2>/dev/null || echo none)"

echo "== 4. Restart $SERVICE"
since=$(date '+%Y-%m-%d %H:%M:%S')
systemctl restart "$SERVICE"
echo "   waiting for the detector to load..."
for _ in $(seq 1 45); do
  if journalctl -u "$SERVICE" --since "$since" --no-pager 2>/dev/null | grep -qE "Camera ready"; then break; fi
  sleep 2
done
sleep 5
journalctl -u "$SERVICE" --since "$since" --no-pager -o cat 2>/dev/null \
  | grep -E "Passive item detection|Item detection|Camera ready|detector:|Background item detection|Traceback|Error" \
  | tail -20 | sed 's/^/   /' || true

echo "== 5. Camera /health"
token=$(grep '^CAMERA_TOKEN=' "$ENVF" | tail -1 | cut -d= -f2- || true)
port=$(grep '^CAMERA_PORT=' "$ENVF" | tail -1 | cut -d= -f2- || true)
curl -s -m 10 -H "X-Camera-Token: $token" "http://127.0.0.1:${port:-8765}/health" | sed 's/^/   /' \
  || echo "   no answer on port ${port:-8765}"
echo
echo "   \"passive\": {\"state\": \"running\"} means sightings are being posted; anything else says why not."
