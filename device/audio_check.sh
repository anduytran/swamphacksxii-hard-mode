#!/usr/bin/env bash
# Audio check for the wearable: Bluetooth speaker link, a C major scale on the output,
# and (with --mic) a short recording played back.
#
#   audio_check.sh            # speaker only
#   audio_check.sh --mic      # also record 3 s and play it back
#
# Devices come from /etc/lastscene/device.env (AUDIO_OUT_DEVICE / AUDIO_IN_DEVICE) when present,
# else bologna / plughw:1,0. The speaker MAC is read from that device's entry in /etc/asound.conf.
# Install as a command (no root; ~/.local/bin is on PATH at login on Raspberry Pi OS):
#   mkdir -p ~/.local/bin && install -m 755 audio_check.sh ~/.local/bin/audiotest
#   install -m 644 play_scale.py ~/.local/bin/play_scale.py
set -u

OUT=bologna
IN=plughw:1,0
if [ -r /etc/lastscene/device.env ]; then
    v=$(sed -n 's/^AUDIO_OUT_DEVICE=//p' /etc/lastscene/device.env | tail -1); [ -n "$v" ] && OUT=$v
    v=$(sed -n 's/^AUDIO_IN_DEVICE=//p' /etc/lastscene/device.env | tail -1); [ -n "$v" ] && [ "$v" != default ] && IN=$v
fi
HERE=$(dirname "$(readlink -f "$0")")
FAILED=0
ok()   { printf '  \033[32mOK\033[0m    %s\n' "$*"; }
bad()  { printf '  \033[31mFAIL\033[0m  %s\n' "$*"; FAILED=1; }
info() { printf '        %s\n' "$*"; }

echo "Output: $OUT    Input: $IN"

# Bluetooth link, if the output is a BlueALSA device in /etc/asound.conf.
MAC=$(awk -v d="pcm.$OUT" '$1==d{f=1} f&&/device/{gsub(/"/,"",$2); print $2; exit}' /etc/asound.conf 2>/dev/null)
if [ -n "$MAC" ]; then
    echo "Bluetooth ($MAC)"
    if ! systemctl is-active -q bluetooth; then bad "bluetooth service not running"; fi
    if ! bluetoothctl show | grep -q "Powered: yes"; then
        info "radio off, turning it on"; sudo rfkill unblock bluetooth 2>/dev/null; bluetoothctl power on >/dev/null
    fi
    if bluetoothctl info "$MAC" 2>/dev/null | grep -q "Connected: yes"; then
        ok "speaker connected"
    else
        info "not connected, connecting (is the speaker on?)"
        bluetoothctl --timeout 15 connect "$MAC" >/dev/null 2>&1; sleep 2
        if bluetoothctl info "$MAC" 2>/dev/null | grep -q "Connected: yes"; then ok "speaker connected"
        else bad "could not connect: turn the speaker on, or re-pair it (device/README.md, Bluetooth speaker)"; fi
    fi
fi

echo "Speaker"
if python3 "$HERE/play_scale.py" -D "$OUT" 2>/dev/null; then ok "played a C major scale on $OUT (did you hear it?)"
else bad "playback on $OUT failed"; fi

if [ "${1:-}" = "--mic" ]; then
    echo "Microphone"
    # The Voice Bonnet's analog mics aren't routed to the recorder after a reboot (unless saved with
    # `sudo alsactl store`); turning the route on needs no root.
    card=$(sed -n 's/^plughw:\([0-9]*\),.*/\1/p' <<<"$IN")
    if [ -n "$card" ] && grep -q voicebonnet "/proc/asound/card$card/id" 2>/dev/null; then
        for s in "Sto1 ADC MIXL ADC1" "Sto1 ADC MIXR ADC1" "RECMIXL BST1" "RECMIXR BST1" "RECMIXL BST2" "RECMIXR BST2"; do
            amixer -c "$card" -q sset "$s" on
        done
        amixer -c "$card" -q sset "IN1 Boost" 4; amixer -c "$card" -q sset "IN2 Boost" 4
    fi
    info "recording 3 s, say something..."
    if arecord -q -D "$IN" -f S16_LE -r 16000 -c 1 -d 3 /tmp/audiotest.wav 2>/dev/null; then
        level=$(python3 -c "
import wave, array, math
w = wave.open('/tmp/audiotest.wav'); a = array.array('h', w.readframes(w.getnframes()))
print(round(math.sqrt(sum(x * x for x in a) / max(len(a), 1))), max(map(abs, a), default=0))")
        set -- $level
        if [ "$2" -eq 0 ]; then bad "recording is silent: mic not routed (device/README.md, Voice Bonnet mic)"
        else ok "recorded (rms $1, peak $2), playing it back"; aplay -q -D "$OUT" /tmp/audiotest.wav 2>/dev/null; fi
    else
        bad "could not record from $IN"
    fi
fi

[ "$FAILED" -eq 0 ] && echo "All good." || { echo "Something failed, see above."; exit 1; }
