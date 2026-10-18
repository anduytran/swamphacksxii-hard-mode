#!/usr/bin/env bash
# Live demo: the real wearable client on the audio Pi (sreer), in the foreground, with the latest code.
# Talk to it ("Last Scene, I'm putting my pin on the desk", "Last Scene, where is my pin?"): every request
# posts live to the API and shows up on the dashboard (Live, Timeline, Items) within seconds.
#
#     device/demo_live.sh [user@host]          default sreer@100.78.33.50; Ctrl-C stops the client
#
# Stops the service copy first (only one program can hold the mic and buttons), pulls the repo on the Pi,
# copies the client, its speaker EQ and sounds into /opt/lastscene, then runs it with /etc/lastscene/device.env.
set -euo pipefail

HOST="${1:-sreer@100.78.33.50}"

ssh -t "$HOST" "sudo systemctl stop lastscene-device 2>/dev/null; \
sudo pkill -f '^/opt/lastscene/venv/bin/python /opt/lastscene/lastscene_device.py'; sleep 1; \
cd ~/lastscene-src && git pull -q && git log --oneline -1 && \
sudo cp -r device/lastscene_device.py device/speaker_eq.py device/sounds /opt/lastscene/ && \
sudo bash -c 'set -a; . /etc/lastscene/device.env; /opt/lastscene/venv/bin/python /opt/lastscene/lastscene_device.py'"
