#!/bin/bash

# =================================================================
# ScriboGenie — Master Installer & Launcher for Raspberry Pi
# =================================================================

# 1. FIND OWN LOCATION
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

# 2. RASPBERRY PI CHECK
is_pi=false
if grep -q "Raspberry Pi" /proc/device-tree/model 2>/dev/null; then
    is_pi=true
else
    echo "⚠️  WARNING: This script is designed for Raspberry Pi."
    echo "   Detected: $(cat /proc/device-tree/model 2>/dev/null || echo "Unknown hardware")"
    read -p "   Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# 3. ENSURE PERMISSIONS
echo "🔧 Setting permissions for scripts..."
chmod +x *.sh 2>/dev/null

# --- FUNCTIONS ---

show_menu() {
    clear
    echo "=================================================="
    echo "         SCRIBOGENIE MASTER CONTROL"
    echo "=================================================="
    echo " 1) [First Time] Full Setup (Install Everything)"
    echo " 2) Launch Application Manually"
    echo " 3) Configure Hotspot Only"
    echo " 4) Check Service Status"
    echo " 5) View Live App Logs"
    echo " 6) Restore WiFi / Remove Hotspot"
    echo " 7) Exit"
    echo "=================================================="
    read -p "Select an option [1-7]: " choice
}

full_setup() {
    echo "🚀 Starting Full Setup..."

    # Install/verify podman + rootless deps (non-interactive, auto-version-checked)
    echo "🐳 Installing/verifying podman..."
    sudo python3 ./podmansetup.py
    if [ $? -ne 0 ]; then
        echo "❌ ERROR: podmansetup.py failed."
        echo "   Run 'sudo python3 ./podmansetup.py --check' to diagnose."
        exit 1
    fi

    # Run Hotspot Setup (host-level WiFi AP, unchanged)
    echo "📡 Configuring WiFi Hotspot..."
    ./setup_hotspot.sh
    if [ $? -ne 0 ]; then
        echo "❌ ERROR: setup_hotspot.sh failed."
        exit 1
    fi

    echo "✅ Setup Complete!"
    echo "--------------------------------------------------"
    echo "📡 HOTSPOT CREDENTIALS:"
    echo "   Network: ScriboGenie"
    echo "   Password: scribogenie"
    echo "   Pi IP: 192.168.4.1"
    echo "--------------------------------------------------"
    echo "Next: build the container (podman build -t scribogenie -f deploy/Containerfile .)"
    echo "      then enable the service (sudo cp scribogenie.service /etc/systemd/system/ && sudo systemctl enable --now scribogenie)."
    echo "⚠️  A REBOOT IS REQUIRED to activate the hotspot and service."
    read -p "Reboot now? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        sudo reboot
    fi
}

manual_launch() {
    echo "🎨 Launching ScriboGenie (podman)..."
    export DISPLAY=:0
    ./start_scribogenie.sh
}

setup_hotspot_only() {
    echo "📡 Configuring WiFi Hotspot..."
    ./setup_hotspot.sh
}

check_status() {
    echo "📊 Service Status:"
    systemctl status scribogenie.service --no-pager
}

restore_wifi() {
    echo "📡 Restoring normal WiFi (removing hotspot)..."
    ./setup_hotspot.sh restore
    read -p "Press Enter to continue..."
}

view_logs() {
    echo "📝 Press Ctrl+C to exit logs..."
    journalctl -u scribogenie.service -f
}

# --- MAIN LOOP ---

while true; do
    show_menu
    case $choice in
        1) full_setup ;;
        2) manual_launch ;;
        3) setup_hotspot_only ;;
        4) check_status; read -p "Press Enter to continue..." ;;
        5) view_logs ;;
        6) restore_wifi ;;
        7) exit 0 ;;
        *) echo "Invalid option"; sleep 1 ;;
    esac
done
