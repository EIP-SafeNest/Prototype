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
    sudo apt install -y python3-pip python3-venv libgl1-mesa-glx libglib2.0-0 git
else
    echo -e "${GREEN}[INFO]${NC} Système non-Linux détecté (macOS?)"
fi

# Create virtual environment
echo -e "${YELLOW}[STEP 2/6]${NC} Création de l'environnement virtuel..."
python3 -m venv venv
source venv/bin/activate

# Install Python packages
echo -e "${YELLOW}[STEP 3/6]${NC} Installation des packages Python..."
pip install --upgrade pip
pip install flask flask-socketio opencv-python-headless ultralytics twilio numpy huggingface_hub

# Download YOLO fire model
echo -e "${YELLOW}[STEP 4/6]${NC} Téléchargement du modèle YOLO fire detection..."

# Check for HuggingFace token
if [ -z "$HF_TOKEN" ]; then
    echo -e "${RED}[ERROR]${NC} Variable HF_TOKEN non définie!"
    echo "Créez un token sur https://huggingface.co/settings/tokens"
    echo "Puis exécutez: export HF_TOKEN='votre_token'"
    exit 1
fi

python3 -c "
from huggingface_hub import hf_hub_download
import shutil
import os

token = os.environ.get('HF_TOKEN')

print('Téléchargement du modèle YOLO11-X fire detection...')
model_path = hf_hub_download(
    repo_id='leeyunjai/yolo11-firedetect',
    filename='firedetect-11x.pt',
    token=token
)
shutil.copy(model_path, 'fire_yolov8.pt')
print('✅ Modèle téléchargé!')
"

# Verify installation
echo -e "${YELLOW}[STEP 5/6]${NC} Vérification de l'installation..."
python3 -c "
from ultralytics import YOLO
model = YOLO('fire_yolov8.pt')
print('Classes:', model.names)
print('✅ Modèle chargé avec succès!')
"

echo -e "${YELLOW}[STEP 6/6]${NC} Installation terminée!"
echo ""
echo "=============================================="
echo -e "${GREEN}✅ Installation réussie!${NC}"
echo "=============================================="
echo ""
echo "Pour lancer l'application:"
echo "  source venv/bin/activate"
echo "  python web_app.py"
echo ""
echo "Puis ouvrez: http://localhost:5001"
echo "=============================================="
