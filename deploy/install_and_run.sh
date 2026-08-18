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
    # No `clear` here: it would erase the previous option's output before the
    # user can read it. The screen is scrolled instead so errors stay visible.
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
    if [ ! -f "$SCRIPT_DIR/scribogenie.service" ]; then
        echo "ERROR: scribogenie.service not found in $SCRIPT_DIR"
        return 1
    fi
    # The service ExecStart points at /usr/local/bin/start_scribogenie.sh
    sudo cp "$SCRIPT_DIR/start_scribogenie.sh" /usr/local/bin/start_scribogenie.sh
    sudo chmod +x /usr/local/bin/start_scribogenie.sh
    sudo cp "$SCRIPT_DIR/scribogenie.service" /etc/systemd/system/
    sudo systemctl daemon-reload
    sudo systemctl enable scribogenie.service
    echo "Service enabled. It will start on next boot."
    read -p "Start it now? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        sudo systemctl start scribogenie.service
    fi
}

ts() {
    echo "[$(date +%H:%M:%S)]"
}

# Returns the installed podman version (e.g. "4.9.3"), or empty if absent.
podman_version() {
    command -v podman >/dev/null 2>&1 || return 1
    podman --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1
}

# True if the container image is already present (either remote ref or local tag).
image_exists() {
    podman image exists "$IMAGE_NAME" 2>/dev/null && return 0
    podman image exists "$IMAGE_TAG" 2>/dev/null
}

full_setup() {
    echo "=================================================="
    echo "  ScriboGenie — Full Setup"
    echo "=================================================="

    # ---- 1/5: Podman ---------------------------------------------------
    echo "$(ts) [Step 1/5] Checking podman..."
    local pv
    pv="$(podman_version)"
    if [ -n "$pv" ] && [ "$(printf '%s\n%s\n' "$pv" "4.3.0" | sort -V | head -1)" = "4.3.0" ]; then
        echo "$(ts)   podman $pv already installed — skipping install"
    else
        echo "$(ts)   podman not installed (or too old) — installing via apt..."
        sudo python3 "$SCRIPT_DIR/podmansetup.py" --verbose
        if [ $? -ne 0 ]; then
            echo "$(ts) ERROR: podmansetup.py failed."
            echo "   Run 'sudo python3 $SCRIPT_DIR/podmansetup.py --check' to diagnose."
            return 1
        fi
        echo "$(ts)   podman installed OK"
    fi
    echo

    # ---- 2/5: Code -----------------------------------------------------
    echo "$(ts) [Step 2/5] Fetching code..."
    if [ -d "$HOME/$REPO_DIR/.git" ]; then
        echo "$(ts)   ~/$REPO_DIR exists — pulling latest from $REPO_BRANCH..."
    else
        echo "$(ts)   no checkout yet — cloning $REPO_BRANCH..."
    fi
    get_code
    echo

    # ---- 3/5: Image ----------------------------------------------------
    echo "$(ts) [Step 3/5] Container image (~1GB)..."
    if image_exists; then
        echo "$(ts)   image already present — skipping pull"
    else
        echo "$(ts)   pulling from $IMAGE_NAME (may take 10-30 min on first run)..."
        pull_image
        if [ $? -ne 0 ]; then
            echo "$(ts) ERROR: image pull failed. Aborting setup."
            return 1
        fi
    fi
    echo

    # ---- 4/5: Hotspot --------------------------------------------------
    echo "$(ts) [Step 4/5] WiFi hotspot..."
    if [ -f /etc/hostapd/hostapd.conf ] && grep -q "ssid=ScriboGenie" /etc/hostapd/hostapd.conf 2>/dev/null; then
        echo "$(ts)   hotspot already configured — skipping"
    else
        echo "$(ts)   configuring hotspot (SSID: ScriboGenie / pass: scribogenie)..."
        "$SCRIPT_DIR/setup_hotspot.sh"
        if [ $? -ne 0 ]; then
            echo "$(ts) ERROR: setup_hotspot.sh failed."
            return 1
        fi
    fi
    echo

    # ---- 5/5: Service --------------------------------------------------
    echo "$(ts) [Step 5/5] Autostart service..."
    if systemctl is-enabled scribogenie.service >/dev/null 2>&1; then
        echo "$(ts)   service already enabled — skipping"
    else
        install_service
    fi
    echo

    echo "$(ts) Setup Complete!"
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

# Pre-flight checks for launching the container. Prints each missing item
# with the exact fix. Returns 0 if all hard requirements pass.
check_launch_prereqs() {
    local ok=0

    echo "Checking launch prerequisites..."

    if ! command -v podman >/dev/null 2>&1; then
        echo "  [x] podman is NOT installed."
        echo "      Fix: choose option 1, or run:  sudo python3 $SCRIPT_DIR/podmansetup.py"
        ok=1
    elif ! podman image exists "$IMAGE_TAG" >/dev/null 2>&1; then
        echo "  [x] container image '$IMAGE_TAG' is not present."
        echo "      Fix: choose option 3 (pull from GHCR) or option 4 (load from file)."
        ok=1
    fi

    if command -v podman >/dev/null 2>&1 && podman ps --format '{{.Names}}' 2>/dev/null | grep -qx 'scribogenie'; then
        echo "  [x] container 'scribogenie' is already running (started by the autostart service?)."
        echo "      Fix:  podman stop scribogenie"
        echo "      Then choose option 5 again."
        ok=1
    fi

    if [ ! -d /tmp/.X11-unix ]; then
        echo "  [x] no X server socket found at /tmp/.X11-unix."
        echo "      Fix: launch from the graphical desktop session (DISPLAY=:0), not a plain SSH shell."
        ok=1
    fi

    if [ ! -f "$HOME/.Xauthority" ]; then
        echo "  [x] XAUTHORITY file missing at $HOME/.Xauthority."
        echo "      Fix: run from the graphical session so the file exists, or copy it from a logged-in desktop."
        ok=1
    fi

    if [ ! -e /dev/snd ]; then
        echo "  [w] /dev/snd missing — audio / TTS will not work."
    fi

    if [ ! -e /dev/input ]; then
        echo "  [w] /dev/input missing — drawing input may not work."
    fi

    return $ok
}

manual_launch() {
    echo "Launching ScriboGenie (podman)..."
    export DISPLAY=:0

    if ! check_launch_prereqs; then
        echo
        echo "  Some prerequisites are missing — nothing was launched."
        echo "  Fix the items above, then try option 5 again."
        return 1
    fi

    echo "  All prerequisites OK. Starting container (Ctrl+C to stop)..."
    "$SCRIPT_DIR/start_scribogenie.sh" 2>&1 | tee -a "$SCRIPT_DIR/launch.log"
    local rc=${PIPESTATUS[0]}
    if [ "$rc" -ne 0 ]; then
        echo
        echo "ERROR: container failed to start (exit $rc)."
        echo "  Last log lines:"
        tail -n 20 "$SCRIPT_DIR/launch.log" | sed 's/^/    /'
        echo
        echo "  Likely fixes:"
        grep -qi "already in use\|name.*scribogenie" "$SCRIPT_DIR/launch.log" \
            && echo "    - container name in use  ->  podman stop scribogenie"
        grep -qi "unable to find image\|short-name\|No such image" "$SCRIPT_DIR/launch.log" \
            && echo "    - image missing  ->  choose option 3 or 4 first"
        grep -qi "cannot find UID/GID\|subuid" "$SCRIPT_DIR/launch.log" \
            && echo "    - rootless subuid issue  ->  sudo python3 $SCRIPT_DIR/podmansetup.py"
        grep -qi "display\|XOpenDisplay\|no display\|XAUTHORITY" "$SCRIPT_DIR/launch.log" \
            && echo "    - display/XAUTHORITY issue  ->  launch from the graphical session"
        grep -qi "permission denied" "$SCRIPT_DIR/launch.log" \
            && echo "    - device mount permission issue  ->  check /dev/snd and /dev/input"
    fi
    return "$rc"
}

setup_hotspot_only() {
    echo "Configuring WiFi Hotspot..."
    "$SCRIPT_DIR/setup_hotspot.sh"
}

check_status() {
    echo "Service Status:"
    systemctl status scribogenie.service --no-pager
}

restore_wifi() {
    echo "Restoring normal WiFi (removing hotspot)..."
    "$SCRIPT_DIR/setup_hotspot.sh" restore
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
        1) full_setup 2>&1 | tee -a "$SCRIPT_DIR/setup.log"; read -p "Press Enter to continue..." ;;
        2) get_code; read -p "Press Enter to continue..." ;;
        3) pull_image; read -p "Press Enter to continue..." ;;
        4) load_image; read -p "Press Enter to continue..." ;;
        5) manual_launch; read -p "Press Enter to continue..." ;;
        6) setup_hotspot_only ;;
        7) install_service; read -p "Press Enter to continue..." ;;
        8) check_status; read -p "Press Enter to continue..." ;;
        9) view_logs ;;
        10) restore_wifi ;;
        11) exit 0 ;;
        *) echo "Invalid option"; sleep 1 ;;
    esac
done
