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

python3 web_app.py
