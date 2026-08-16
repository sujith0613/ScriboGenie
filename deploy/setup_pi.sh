#!/bin/bash
# setup_pi.sh
# LEGACY: installs the OLD venv + TensorFlow/espeak stack for the legacy app_pi.py.
#
# SUPERSEDED — the current deployment runs the modern ONNX + KittenTTS app in a
# self-contained podman container. Use these instead:
#   sudo python3 ./podmansetup.py      # install/verify podman + rootless deps
#   podman build -t scribogenie -f ./Containerfile .   # build the image
#   ./setup_hotspot.sh                 # host-level WiFi AP
# Kept for reference / legacy Pi repo compatibility.

echo "=================================================="
echo "  ScriboGenie — High-Performance Pi Installer"
echo "=================================================="

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# 1. System updates
sudo apt-get update
sudo apt-get install -y python3-pip python3-venv python3-tk
sudo apt-get install -y libatlas-base-dev libopencv-dev
sudo apt-get install -y espeak libinput-dev evdev ghostscript

# 2. Create Virtual Environment
python3 -m venv "$SCRIPT_DIR/scribogenie_env"
source "$SCRIPT_DIR/scribogenie_env/bin/activate"

# 3. Install Python packages with specific versions for Pi 4
"$SCRIPT_DIR/scribogenie_env/bin/pip" install --upgrade pip
"$SCRIPT_DIR/scribogenie_env/bin/pip" install "numpy>=1.23.5,<2.0.0" 
"$SCRIPT_DIR/scribogenie_env/bin/pip" install "keras>=3.0.0"
"$SCRIPT_DIR/scribogenie_env/bin/pip" install "tensorflow==2.15.0"
"$SCRIPT_DIR/scribogenie_env/bin/pip" install pyspellchecker websockets evdev pillow

# 4. Generate systemd service with correct path
cat > /tmp/scribogenie.service << EOF
[Unit]
Description=ScriboGenie Handwriting App
After=graphical.target

[Service]
Type=simple
ExecStart=$SCRIPT_DIR/start_scribogenie.sh
Restart=always
RestartSec=10
User=$USER

[Install]
WantedBy=graphical.target
EOF

sudo mv /tmp/scribogenie.service /etc/systemd/system/scribogenie.service
sudo systemctl daemon-reload

# 5. Fix permissions for Wacom
sudo usermod -a -G input $USER

echo "=================================================="
echo "  Installation Complete! ✅"
echo "  Please REBOOT your Pi before running the app."
echo "=================================================="