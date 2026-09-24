#!/usr/bin/env bash
# =============================================================
# 📡 SafeNest — Partage de caméra (Mac et Linux)
#
#   ./lancer_camera.sh                  # caméra 0, port 8081
#   ./lancer_camera.sh --camera 1       # autre caméra
#   ./lancer_camera.sh --name "Salon"   # nom affiché dans SafeNest
#
# Au premier lancement, installe tout seul ce qu'il faut
# dans .venv-camera/ à côté du script.
# =============================================================
set -e
cd "$(dirname "$0")"

VENV=".venv-camera"

if ! command -v python3 >/dev/null 2>&1; then
    echo "[ERREUR] python3 introuvable."
    echo "  Mac   : brew install python"
    echo "  Linux : sudo apt install python3 python3-venv"
    exit 1
fi

if [ ! -x "$VENV/bin/python" ]; then
    echo "[INSTALL] Première utilisation : création de l'environnement..."
    if ! python3 -m venv "$VENV"; then
        rm -rf "$VENV"
        echo "[ERREUR] Impossible de créer l'environnement."
        echo "  Linux : sudo apt install python3-venv"
        exit 1
    fi
fi

if ! "$VENV/bin/python" -c "import cv2" 2>/dev/null; then
    echo "[INSTALL] Installation d'OpenCV (une seule fois, ~1 min)..."
    if ! "$VENV/bin/python" -m pip install -q --disable-pip-version-check opencv-python-headless; then
        echo "[ERREUR] L'installation d'OpenCV a échoué (vérifie ta connexion Internet)."
        exit 1
    fi
fi

echo "[OK] Lancement du partage de caméra — Ctrl+C pour arrêter"
exec "$VENV/bin/python" mac_camera_stream.py "$@"
