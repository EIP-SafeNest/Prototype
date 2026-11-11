#!/usr/bin/env bash
# Installe SafeNest comme service sur le Raspberry Pi :
#   - ffmpeg (clips vidéo lisibles sur iPhone, Android et navigateur)
#   - avahi (le boîtier répond à http://<nom-du-pi>.local, ex. http://safenest.local)
#   - service systemd « safenest » lancé au démarrage, relancé en cas de plantage
#
# À lancer depuis le dossier Prototype, avec un environnement virtuel déjà créé (install.sh) :
#   ./deploy/install_service.sh
set -euo pipefail

APP_DIR="$(cd "$(dirname "$0")/.." && pwd)"
APP_USER="$(id -un)"
CONFIG_DIR="$HOME/.config/safenest"
PYTHON="$APP_DIR/venv/bin/python"

# Installation sans terminal (ex. via ssh) : SUDO_ASKPASS=<script qui affiche le mot de passe>
SUDO=(sudo)
if [ -n "${SUDO_ASKPASS:-}" ]; then SUDO=(sudo -A); fi

if [ ! -x "$PYTHON" ]; then
    echo "Environnement virtuel introuvable ($PYTHON) : lancez d'abord ./install.sh" >&2
    exit 1
fi

echo "==> Paquets système (ffmpeg, avahi)"
"${SUDO[@]}" apt-get update -qq
"${SUDO[@]}" env DEBIAN_FRONTEND=noninteractive apt-get install -y -qq ffmpeg avahi-daemon
# safenest-slim.sh masque avahi par défaut (sauf --keep-avahi) : on le réactive pour safenest.local
if [[ "$(systemctl is-enabled avahi-daemon.service 2>/dev/null || true)" == masked* ]]; then
    echo "    avahi était masqué (safenest-slim.sh) : réactivation pour l'adresse .local"
    "${SUDO[@]}" systemctl unmask avahi-daemon.service avahi-daemon.socket
fi
"${SUDO[@]}" systemctl enable --now avahi-daemon

echo "==> Dépendances Python du boîtier"
# requests sert à joindre l'API (install.sh ne l'installe pas explicitement)
"$PYTHON" -m pip install -q requests

echo "==> Configuration ($CONFIG_DIR/env)"
mkdir -p "$CONFIG_DIR"
chmod 700 "$CONFIG_DIR"
if [ ! -f "$CONFIG_DIR/env" ]; then
    cp "$APP_DIR/deploy/safenest.env.example" "$CONFIG_DIR/env"
    echo "    Fichier créé : vérifiez SAFENEST_API_URL dans $CONFIG_DIR/env"
else
    echo "    Fichier existant conservé"
fi

echo "==> Service systemd"
"${SUDO[@]}" tee /etc/systemd/system/safenest.service >/dev/null <<EOF
[Unit]
Description=SafeNest - détection locale et alertes
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=$APP_USER
WorkingDirectory=$APP_DIR
EnvironmentFile=-$CONFIG_DIR/env
Environment=PYTHONUNBUFFERED=1
ExecStart=$PYTHON $APP_DIR/web_app_rpi.py
Restart=always
RestartSec=5
# Permet d'écouter sur le port 80 sans être root
AmbientCapabilities=CAP_NET_BIND_SERVICE
NoNewPrivileges=true

[Install]
WantedBy=multi-user.target
EOF

"${SUDO[@]}" systemctl daemon-reload
"${SUDO[@]}" systemctl enable safenest
"${SUDO[@]}" systemctl restart safenest

echo
echo "SafeNest est lancé. Interface : http://$(hostname | tr '[:upper:]' '[:lower:]').local"
echo "Journal : journalctl -u safenest -f"
