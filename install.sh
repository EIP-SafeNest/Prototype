#!/bin/bash
# =============================================================
# 🔥 Fire Detection System - Installation Script for Raspberry Pi
# =============================================================

set -e  # Exit on error

echo "=============================================="
echo "🔥 Fire Detection System - Installation"
echo "=============================================="

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if running on Raspberry Pi or Linux
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo -e "${GREEN}[INFO]${NC} Système Linux détecté"
    
    # Install system dependencies
    echo -e "${YELLOW}[STEP 1/6]${NC} Installation des dépendances système..."
    sudo apt update
    sudo apt install -y python3-pip python3-venv libglib2.0-0 git
    # Try to install OpenGL libs (may vary by distro)
    sudo apt install -y libgl1 || sudo apt install -y libgl1-mesa-dev || echo "OpenGL libs skipped"
else
    echo -e "${GREEN}[INFO]${NC} Système non-Linux détecté (macOS?)"
fi

# Create virtual environment
echo -e "${YELLOW}[STEP 2/6]${NC} Création de l'environnement virtuel..."
python3 -m venv venv
source venv/bin/activate

# Install Python packages
echo -e "${YELLOW}[STEP 3/4]${NC} Installation des packages Python..."
pip install --upgrade pip

# Detect architecture
ARCH=$(uname -m)
echo -e "${GREEN}[INFO]${NC} Architecture détectée: $ARCH"

if [[ "$ARCH" == "aarch64" ]] || [[ "$ARCH" == "armv7l" ]]; then
    # Raspberry Pi ARM - lightweight version (NO PyTorch)
    echo -e "${YELLOW}[INFO]${NC} Mode Raspberry Pi: installation légère sans PyTorch"
    pip install flask flask-socketio opencv-python-headless twilio numpy
    
    # Set flag for run script
    echo "RPI_MODE=1" > .rpi_mode
    echo -e "${GREEN}[INFO]${NC} Détection feu par analyse couleur (optimisé ARM)"
else
    # x86/x64 - full version with YOLO
    echo -e "${YELLOW}[INFO]${NC} Installation complète avec YOLO..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    pip install flask flask-socketio opencv-python-headless ultralytics twilio numpy huggingface_hub
    
    # Download YOLO fire model only on x86
    if [ -n "$HF_TOKEN" ]; then
        echo -e "${YELLOW}[STEP 4/4]${NC} Téléchargement du modèle YOLO fire detection..."
        python3 -c "
from huggingface_hub import hf_hub_download
import shutil
import os

token = os.environ.get('HF_TOKEN')
# Options: firedetect-11n.pt (nano/rapide), firedetect-11s.pt (small/équilibré), 
#          firedetect-11m.pt (medium), firedetect-11x.pt (extra-large/précis mais lent)
print('Téléchargement du modèle YOLO11-S fire detection (équilibré vitesse/précision)...')
model_path = hf_hub_download(
    repo_id='leeyunjai/yolo11-firedetect',
    filename='firedetect-11s.pt',
    token=token
)
shutil.copy(model_path, 'fire_yolov8.pt')
print('✅ Modèle téléchargé!')
"
        # Verify
        python3 -c "
from ultralytics import YOLO
model = YOLO('fire_yolov8.pt')
print('Classes:', model.names)
print('✅ Modèle YOLO chargé!')
"
    else
        echo -e "${YELLOW}[INFO]${NC} HF_TOKEN non défini - modèle YOLO standard"
    fi
fi

echo ""
echo "=============================================="
echo -e "${GREEN}✅ Installation réussie!${NC}"
echo "=============================================="
echo ""
echo "Pour lancer: ./run.sh"
echo "Interface: http://localhost:5001"
echo "=============================================="
