#!/bin/bash

# =================================================================
# ScriboGenie — Master Installer & Launcher for Raspberry Pi
# Self-contained podman deployment. App, models, and KittenTTS cache
# are baked into a container image pulled from GHCR (public, free).
# =================================================================

# Repo/image identifiers
REPO_URL="https://github.com/sujith0613/ScriboGenie.git"
REPO_BRANCH="context-brain"
REPO_DIR="scribo"                # canonical checkout dir on the Pi (~/$REPO_DIR)
IMAGE_NAME="ghcr.io/sujith0613/scribogenie:arm64"
IMAGE_TAG="scribogenie:latest"   # local alias the service/launcher uses

# 1. FIND OWN LOCATION
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

# 2. RASPBERRY PI CHECK
is_pi=false
if grep -q "Raspberry Pi" /proc/device-tree/model 2>/dev/null; then
    is_pi=true
else
    echo "WARNING: This script is designed for Raspberry Pi."
    echo "   Detected: $(cat /proc/device-tree/model 2>/dev/null || echo "Unknown hardware")"
    read -p "   Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# 3. ENSURE PERMISSIONS
echo "Setting permissions for scripts..."
chmod +x *.sh 2>/dev/null

# --- FUNCTIONS ---

show_menu() {
    clear
    echo "=================================================="
    echo "         SCRIBOGENIE MASTER CONTROL"
    echo "=================================================="
    echo " 1) [First Time] Full Setup (Everything)"
    echo " 2) Get / Update Code (git)"
    echo " 3) Pull Container Image (GHCR)"
    echo " 4) Load Image From File (offline fallback)"
    echo " 5) Launch Application Manually"
    echo " 6) Configure Hotspot Only"
    echo " 7) Install / Enable Autostart Service"
    echo " 8) Check Service Status"
    echo " 9) View Live App Logs"
    echo "10) Restore WiFi / Remove Hotspot"
    echo "11) Exit"
    echo "=================================================="
    read -p "Select an option [1-11]: " choice
}

get_code() {
    echo "Fetching ScriboGenie code into ~/$REPO_DIR..."
    cd "$HOME"
    if [ -d "$REPO_DIR" ]; then
        echo "   Repo exists — pulling latest..."
        cd "$REPO_DIR" && git pull --rebase origin "$REPO_BRANCH"
    else
        git clone -b "$REPO_BRANCH" "$REPO_URL" "$REPO_DIR"
        cd "$REPO_DIR"
    fi
    if [ $? -ne 0 ]; then
        echo "ERROR: failed to fetch code. Is the Pi online?"
        return 1
    fi
    chmod +x deploy/*.sh 2>/dev/null
    echo "Code up to date."
}

pull_image() {
    echo "Pulling container image from GHCR..."
    podman pull "$IMAGE_NAME"
    if [ $? -ne 0 ]; then
        echo "ERROR: pull failed. Is the Pi online? Is the package public?"
        return 1
    fi
    podman tag "$IMAGE_NAME" "$IMAGE_TAG"
    echo "Image pulled and tagged as $IMAGE_TAG."
}

load_image() {
    echo "Loading image from file (offline transfer)..."
    read -p "   Path to image tarball (.tar.gz): " tarball
    if [ ! -f "$tarball" ]; then
        echo "ERROR: file not found: $tarball"
        return 1
    fi
    zcat "$tarball" | podman load
    if [ $? -ne 0 ]; then
        echo "ERROR: podman load failed."
        return 1
    fi
    podman tag ghcr.io/sujith0613/scribogenie:arm64 "$IMAGE_TAG" 2>/dev/null
    echo "Image loaded."
}

install_service() {
    echo "Installing autostart service..."
    if [ ! -f scribogenie.service ]; then
        echo "ERROR: scribogenie.service not found in $SCRIPT_DIR"
        return 1
    fi
    # The service ExecStart points at /usr/local/bin/start_scribogenie.sh
    sudo cp start_scribogenie.sh /usr/local/bin/start_scribogenie.sh
    sudo chmod +x /usr/local/bin/start_scribogenie.sh
    sudo cp scribogenie.service /etc/systemd/system/
    sudo systemctl daemon-reload
    sudo systemctl enable scribogenie.service
    echo "Service enabled. It will start on next boot."
    read -p "Start it now? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        sudo systemctl start scribogenie.service
    fi
}

full_setup() {
    echo "Starting Full Setup..."

    echo "Installing/verifying podman..."
    sudo python3 ./podmansetup.py
    if [ $? -ne 0 ]; then
        echo "ERROR: podmansetup.py failed."
        echo "   Run 'sudo python3 ./podmansetup.py --check' to diagnose."
        return 1
    fi

    get_code

    pull_image
    if [ $? -ne 0 ]; then
        echo "ERROR: image pull failed. Aborting setup."
        return 1
    fi

    echo "Configuring WiFi Hotspot..."
    ./setup_hotspot.sh
    if [ $? -ne 0 ]; then
        echo "ERROR: setup_hotspot.sh failed."
        return 1
    fi

    install_service

    echo "Setup Complete!"
    echo "--------------------------------------------------"
    echo "   Code:     ~/$REPO_DIR"
    echo "   Image:    $IMAGE_NAME"
    echo "   Hotspot:  ScriboGenie / scribogenie @ 192.168.4.1"
    echo "--------------------------------------------------"
    echo "A REBOOT IS REQUIRED to activate the hotspot and service."
    read -p "Reboot now? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        sudo reboot
    fi
}

manual_launch() {
    echo "Launching ScriboGenie (podman)..."
    export DISPLAY=:0
    ./start_scribogenie.sh
}

setup_hotspot_only() {
    echo "Configuring WiFi Hotspot..."
    ./setup_hotspot.sh
}

check_status() {
    echo "Service Status:"
    systemctl status scribogenie.service --no-pager
}

restore_wifi() {
    echo "Restoring normal WiFi (removing hotspot)..."
    ./setup_hotspot.sh restore
    read -p "Press Enter to continue..."
}

view_logs() {
    echo "Press Ctrl+C to exit logs..."
    journalctl -u scribogenie.service -f
}

# --- MAIN LOOP ---

while true; do
    show_menu
    case $choice in
        1) full_setup ;;
        2) get_code; read -p "Press Enter to continue..." ;;
        3) pull_image; read -p "Press Enter to continue..." ;;
        4) load_image; read -p "Press Enter to continue..." ;;
        5) manual_launch ;;
        6) setup_hotspot_only ;;
        7) install_service; read -p "Press Enter to continue..." ;;
        8) check_status; read -p "Press Enter to continue..." ;;
        9) view_logs ;;
        10) restore_wifi ;;
        11) exit 0 ;;
        *) echo "Invalid option"; sleep 1 ;;
    esac
done
