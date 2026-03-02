#!/bin/bash
# =============================================================
# 🔥 Fire Detection System - Run Script
# =============================================================

cd "$(dirname "$0")"

# Activate virtual environment
source venv/bin/activate

echo "=============================================="
echo "🔥 Fire Detection System"
echo "=============================================="
echo "Interface: http://localhost:5001"
echo "=============================================="

# Check if running on Raspberry Pi (ARM mode)
if [ -f ".rpi_mode" ]; then
    echo "Mode: Raspberry Pi (détection couleur)"
    python3 web_app_rpi.py
else
    echo "Mode: Standard (YOLO)"
    python3 web_app.py
fi
