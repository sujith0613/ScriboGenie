#!/bin/bash
# start_scribogenie.sh
# Launches ScriboGenie inside its podman container.
# Used by scribogenie.service (systemd autostart on the Pi) and for manual runs.
#
# Requires: podman, an X display (Waveshare 5" LCD), /dev/snd (audio),
#           and /dev/input (Wacom). See deploy/podmansetup.py + deploy/Containerfile.

IMAGE=${SCRIBOGENIE_IMAGE:-scribogenie:latest}
DISPLAY_NUM=${DISPLAY:-:0}
LOG_DIR="${HOME}/scribo/logs"
DATA_DIR="${HOME}/scribo/data"

mkdir -p "$LOG_DIR" "$DATA_DIR"

exec podman run --rm \
    --name scribogenie \
    --network host \
    -e DISPLAY="$DISPLAY_NUM" \
    -v /tmp/.X11-unix:/tmp/.X11-unix:ro \
    -e XAUTHORITY="$HOME/.Xauthority" \
    -v "$HOME/.Xauthority:$HOME/.Xauthority:ro" \
    -v /dev/snd:/dev/snd:ro \
    -v /dev/input:/dev/input:ro \
    -v "$LOG_DIR":/app/logs \
    -v "$DATA_DIR":/app/data \
    "$IMAGE"