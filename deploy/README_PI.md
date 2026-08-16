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
| `install_and_run.sh` | Master menu: Full Setup / Get Code / Pull Image / Load / Launch / Hotspot / Service / Status / Logs / Restore. |
| `start_scribogenie.sh` | Launches the app in podman (used by systemd + manual). |
| `scribogenie.service` | Systemd autostart unit (deployed to `/usr/local/bin/start_scribogenie.sh` + `/etc/systemd/system/`). |

> `setup_pi.sh` is the **legacy** TF/venv installer, kept only for reference.

## Prerequisites

- Raspberry Pi 4 (aarch64), Raspberry Pi OS **64-bit** (Debian 13 Trixie).
- Display: Waveshare 5" HDMI LCD (800x480). Wacom CTL-100WL via USB (evdev).
- Audio: 3.5mm jack or HDMI (KittenTTS output via sounddevice).

## One-time setup (on the Pi) — via GHCR (recommended)

The arm64 image is pre-built and hosted publicly at
`ghcr.io/sujith0613/scribogenie:arm64` (free, unlimited pulls, no Pi credentials).

```bash
# Clone the repo (small — models are baked into the image, not in git)
git clone -b context-brain https://github.com/sujith0613/ScriboGenie.git
cd ScriboGenie

# Run the master menu and pick option 1 (Full Setup):
#   podman install → get code → pull GHCR image → hotspot → autostart service → reboot
./deploy/install_and_run.sh
```

The menu is idempotent — re-run it any time to update code (`git pull`), re-pull the
image, check status, tail logs, or tear down the hotspot.

### Individual steps (what the menu automates)

```bash
# 1. Install podman + rootless deps (idempotent, safe to re-run)
sudo python3 deploy/podmansetup.py --check      # pre-flight (optional)
sudo python3 deploy/podmansetup.py

# 2. Pull the pre-built arm64 image from GHCR and alias it locally
podman pull ghcr.io/sujith0613/scribogenie:arm64
podman tag ghcr.io/sujith0613/scribogenie:arm64 scribogenie:latest

# 3. Configure the host WiFi hotspot (ScriboGenie / scribogenie @ 192.168.4.1)
sudo bash deploy/setup_hotspot.sh

# 4. Autostart
sudo cp deploy/start_scribogenie.sh /usr/local/bin/  && sudo chmod +x /usr/local/bin/start_scribogenie.sh
sudo cp deploy/scribogenie.service /etc/systemd/system/
sudo systemctl enable --now scribogenie
```

## Offline / air-gapped transfer (fallback)

If the Pi has no internet during setup, build or transfer the image elsewhere:

```bash
# From the build machine (laptop): cross-build arm64, then ship the tarball
podman build --platform linux/arm64 -t scribogenie:arm64 -f deploy/Containerfile .
podman save scribogenie:arm64 | gzip > scribogenie-arm64.tar.gz
# copy scribogenie-arm64.tar.gz to the Pi (USB stick / scp / Tailscale)

# On the Pi: load it (menu option 4 does this)
zcat scribogenie-arm64.tar.gz | podman load
podman tag ghcr.io/sujith0613/scribogenie:arm64 scribogenie:latest
```

> The image is large (~2.5 GB gzipped). GHCR is faster and simpler when the Pi has
> network access; use the tarball route only when it doesn't.

## Rebuilding the image (laptop, after code changes)

```bash
git push origin context-brain                 # code first
podman build --platform linux/arm64 -t ghcr.io/sujith0613/scribogenie:arm64 -f deploy/Containerfile .
podman login ghcr.io -u sujith0613            # PAT with write:packages
podman push ghcr.io/sujith0613/scribogenie:arm64
# On the Pi: ./deploy/install_and_run.sh → 3) Pull Container Image (GHCR)
```

> Large model weights are gitignored (`models/*/model*.onnx`) — they exist on the
> build machine and get baked into the image. The Pi never needs them in git.

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