# ScriboGenie — Raspberry Pi Deployment (podman container)

Everything you need to run the **modern** ScriboGenie (ONNX recognizer + KittenTTS)
on a Raspberry Pi as a self-contained container with **zero host runtime deps** —
no Python venv, no TensorFlow, no apt `espeak`, no manual model download.

## Why containerize

- **Self-contained image**: app + recognizer/LM models + KittenTTS cache are all
  baked in (`deploy/Containerfile`). The image runs fully **offline**.
- **Lighter than the legacy stack**: ONNX + KittenTTS replace TensorFlow 2.15 +
  system espeak, so the image needs no TF and no apt espeak.
- **Reproducible**: identical runtime on Pi, laptop, or server (amd64/arm64).

## Files

| File | Purpose |
|---|---|
| `Containerfile` | Builds the self-contained image (bakes models + TTS cache). |
| `podmansetup.py` | Installs/verifies podman + rootless deps, non-interactive. |
| `setup_hotspot.sh` | Host-level WiFi AP (`ScriboGenie` / `scribogenie` @ `192.168.4.1`). `restore` reverses. |
| `install_and_run.sh` | Menu: Full Setup / Launch / Hotspot / Status / Logs / Restore. |
| `start_scribogenie.sh` | Launches the app in podman (used by systemd + manual). |
| `scribogenie.service` | Systemd autostart unit (path-replaced; expects `start_scribogenie.sh` in PATH). |

> `setup_pi.sh` is the **legacy** TF/venv installer, kept only for reference.

## Prerequisites

- Raspberry Pi 4 (aarch64), Raspberry Pi OS **64-bit** (Debian 13 Trixie).
- Display: Waveshare 5" HDMI LCD (800x480). Wacom CTL-100WL via USB (evdev).
- Audio: 3.5mm jack or HDMI (KittenTTS output via sounddevice).

## One-time setup (on the Pi)

```bash
# 1. Install podman + rootless deps (idempotent, safe to re-run)
sudo python3 deploy/podmansetup.py --check      # pre-flight (optional)
sudo python3 deploy/podmansetup.py

# 2. Configure the host WiFi hotspot (ScriboGenie / scribogenie @ 192.168.4.1)
sudo bash deploy/setup_hotspot.sh

# 3. Build the container (do this ON the Pi, or build elsewhere and transfer)
podman build -t scribogenie:latest -f deploy/Containerfile .

#    Transfer from another machine:
#      podman save scribogenie:latest | gzip > scribogenie.tar.gz
#      (on Pi) podman load < scribogenie.tar.gz
```

## Run

```bash
# Manual
export DISPLAY=:0
bash deploy/start_scribogenie.sh

# Autostart (systemd)
sudo cp deploy/scribogenie.service /etc/systemd/system/
sudo systemctl enable --now scribogenie
journalctl -u scribogenie -f
```

Then connect the phone to the **ScriboGenie** WiFi and open
`http://192.168.4.1:8000/` (WebSocket at `ws://192.168.4.1:8765`).

## What `start_scribogenie.sh` passes to the container

| Host | Container | Reason |
|---|---|---|
| `/tmp/.X11-unix` + `$XAUTHORITY` | display | 800x480 GUI |
| `/dev/snd` | audio | KittenTTS playback |
| `/dev/input` | input | Wacom evdev |
| `$HOME/ScriboGenie/logs` | `/app/logs` | persistence |
| `$HOME/ScriboGenie/data` | `/app/data` | progress + corpus |
| `--network host` | — | phone reaches `192.168.4.1:8765/8000` |

## Troubleshooting

- **No audio** → confirm `-v /dev/snd:/dev/snd` is mounted and the speaker works:
  `podman exec scribogenie python -c "import sounddevice as sd; print(sd.query_devices())"`.
- **Wacom not drawing** → verify the tablet shows in `evtest`; the container reads
  `/dev/input` (host user must be in the `input` group).
- **Phone can't connect** → confirm hotspot up (`iw dev wlan0 info`), phone on
  ScriboGenie, `ping 192.168.4.1`.
- **Rebuild after app changes** → the `COPY` steps are cached; only changed files
  re-copy, so rebuilds are fast.