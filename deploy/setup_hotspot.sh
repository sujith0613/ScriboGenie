#!/bin/bash
# setup_hotspot.sh
# Configures Raspberry Pi as a WiFi Hotspot for ScriboGenie
# Usage: ./setup_hotspot.sh           — Configure hotspot
#        ./setup_hotspot.sh restore   — Restore normal WiFi

echo "=================================================="
echo "  ScriboGenie — Hotspot Configuration"
echo "=================================================="

if [ "$1" = "restore" ]; then
    echo "Restoring normal WiFi..."
    sudo rm -f /etc/systemd/network/12-wlan0.network
    sudo nmcli dev set wlan0 managed yes
    sudo systemctl stop hostapd dnsmasq 2>/dev/null
    sudo systemctl disable hostapd dnsmasq 2>/dev/null
    sudo systemctl restart NetworkManager
    echo "WiFi restored. Reboot recommended."
    exit 0
fi

# 1. Install dependencies
sudo apt-get update
sudo apt-get install -y hostapd dnsmasq

# 2. Stop services to configure
sudo systemctl stop hostapd
sudo systemctl stop dnsmasq

# 3. Tell NetworkManager to ignore wlan0 (Debian Trixie+)
echo "Disabling NetworkManager management for wlan0..."
sudo nmcli dev set wlan0 managed no

# 4. Set static IP for wlan0 via systemd-networkd
echo "Setting static IP 192.168.4.1/24 on wlan0..."
sudo bash -c 'cat > /etc/systemd/network/12-wlan0.network << EOF
[Match]
Name=wlan0

[Network]
Address=192.168.4.1/24
EOF'
sudo systemctl enable systemd-networkd
sudo systemctl restart systemd-networkd

# 5. Configure DHCP server (dnsmasq)
sudo mv /etc/dnsmasq.conf /etc/dnsmasq.conf.orig 2>/dev/null
sudo bash -c 'cat << EOF > /etc/dnsmasq.conf
interface=wlan0
dhcp-range=192.168.4.2,192.168.4.20,255.255.255.0,24h
domain=wlan
address=/scribogenie.local/192.168.4.1
EOF'

# 6. Configure Access Point (hostapd)
sudo bash -c 'cat << EOF > /etc/hostapd/hostapd.conf
interface=wlan0
driver=nl80211
ssid=ScriboGenie
hw_mode=g
channel=7
wmm_enabled=0
macaddr_acl=0
auth_algs=1
ignore_broadcast_ssid=0
wpa=2
wpa_passphrase=scribogenie
wpa_key_mgmt=WPA-PSK
wpa_pairwise=TKIP
rsn_pairwise=CCMP
EOF'

# 7. Point hostapd to config file
sudo sed -i 's/#DAEMON_CONF=""/DAEMON_CONF="\/etc\/hostapd\/hostapd.conf"/' /etc/default/hostapd

# 8. Enable and start services
sudo systemctl unmask hostapd
sudo systemctl enable hostapd
sudo systemctl start hostapd
sudo systemctl enable dnsmasq
sudo systemctl start dnsmasq

echo "=================================================="
echo "  Hotspot Ready! ✅"
echo "  SSID: ScriboGenie"
echo "  Pass: scribogenie"
echo "  Pi IP: 192.168.4.1"
echo "  Run './setup_hotspot.sh restore' to undo"
echo "=================================================="
