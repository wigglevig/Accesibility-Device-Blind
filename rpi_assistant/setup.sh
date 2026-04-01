#!/bin/bash
# ============================================================
# Smart Vision Assistant — Raspberry Pi Setup Script
# ============================================================
# Tested on: Raspberry Pi 4B (8GB), Raspberry Pi OS 64-bit (Bookworm)
#
# Usage:
#   chmod +x rpi_assistant/setup.sh
#   ./rpi_assistant/setup.sh
#
# What this does:
#   1. Installs system-level packages (audio, camera, dev tools)
#   2. Creates a Python virtual environment
#   3. Installs Python dependencies (PyTorch, YOLO, etc.)
#   4. Downloads the YOLOv8 nano model
#   5. Tests audio output
#   6. Creates a systemd service for auto-start on boot
# ============================================================

set -e

echo "============================================"
echo "  Smart Vision Assistant — RPi Setup"
echo "============================================"
echo ""

# Determine project root (parent of the directory containing this script)
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$PROJECT_DIR"

echo "Project directory: $PROJECT_DIR"
echo ""

# ----- Step 1: System dependencies -----
echo "[1/6] Installing system dependencies..."
sudo apt-get update -qq

# Core dependencies
sudo apt-get install -y -qq \
    python3-pip \
    python3-venv \
    python3-dev \
    git \
    espeak \
    espeak-ng \
    flac \
    libatlas-base-dev \
    libhdf5-dev \
    libharfbuzz-dev \
    libwebp-dev \
    libjpeg-dev \
    libpng-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    v4l-utils \
    2>/dev/null || true

# PortAudio headers — needed to build pyaudio from source
# Package name varies across Debian/RPi OS versions
echo "  Installing PortAudio dev headers..."
sudo apt-get install -y portaudio19-dev 2>/dev/null || \
    sudo apt-get install -y libportaudio2 libportaudiocpp0 portaudio19-dev 2>/dev/null || \
    sudo apt-get install -y libportaudio-dev 2>/dev/null || \
    echo "  WARNING: Could not install portaudio headers. PyAudio may fail."

# These may not exist on Bookworm — install if available, skip if not
sudo apt-get install -y -qq libtiff5 2>/dev/null || \
    sudo apt-get install -y -qq libtiff-dev 2>/dev/null || true
sudo apt-get install -y -qq libopenjp2-7 2>/dev/null || true

echo "  Done."

# ----- Step 2: Virtual environment -----
echo ""
echo "[2/6] Creating Python virtual environment..."

if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "  Created."
else
    echo "  Already exists, skipping."
fi

source venv/bin/activate
pip install --upgrade pip -q

# ----- Step 3: Python packages -----
echo ""
echo "[3/6] Installing Python packages..."
echo "  This will take 5-15 minutes on first run."
echo ""

# PyTorch — aarch64 wheels available via pip on 64-bit RPi OS
pip install -q torch torchvision torchaudio 2>/dev/null || \
    pip install -q torch torchvision 2>/dev/null || \
    echo "  WARNING: PyTorch install failed. Try manually: pip install torch torchvision"

# Core packages (without pyaudio — installed separately below)
pip install -q \
    ultralytics \
    opencv-python-headless \
    numpy \
    Pillow \
    pyttsx3 \
    SpeechRecognition \
    google-genai \
    python-dotenv

# PyAudio — needs portaudio headers, install separately so failure doesn't break everything
echo "  Installing PyAudio..."
pip install pyaudio 2>/dev/null || \
    pip install --no-build-isolation pyaudio 2>/dev/null || \
    echo "  WARNING: PyAudio install failed. Microphone input may not work."
echo "  If PyAudio failed, run: sudo apt-get install portaudio19-dev && pip install pyaudio"

# RPi.GPIO — usually pre-installed, but ensure it's in the venv
pip install -q RPi.GPIO 2>/dev/null || true

echo "  Done."

# ----- Step 4: YOLO model -----
echo ""
echo "[4/6] Downloading YOLOv8 nano model..."
python3 -c "
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
print('  Done.')
"

# ----- Step 5: Audio test -----
echo ""
echo "[5/6] Testing audio output..."
python3 -c "
import pyttsx3
engine = pyttsx3.init()
engine.setProperty('rate', 160)
engine.say('Audio test. Vision assistant setup is complete.')
engine.runAndWait()
print('  Done.')
" 2>/dev/null || echo "  WARNING: Audio test failed. Check speaker/headphone connection."

# ----- Step 6: Systemd service -----
echo ""
echo "[6/6] Setting up auto-start service..."

# Read Gemini API key from .env if it exists
GEMINI_KEY=""
if [ -f "$PROJECT_DIR/.env" ]; then
    GEMINI_KEY=$(grep -E "^GEMINI_API_KEY=" "$PROJECT_DIR/.env" 2>/dev/null | cut -d'=' -f2- | tr -d "'\"" || true)
fi

SERVICE_FILE="/etc/systemd/system/vision-assistant.service"

sudo tee "$SERVICE_FILE" > /dev/null << EOF
[Unit]
Description=Smart Vision Assistant
After=multi-user.target sound.target
Wants=sound.target

[Service]
Type=simple
User=$(whoami)
WorkingDirectory=$PROJECT_DIR
ExecStart=$PROJECT_DIR/venv/bin/python -m rpi_assistant.main
Restart=on-failure
RestartSec=5
Environment=GEMINI_API_KEY=${GEMINI_KEY}

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable vision-assistant.service
echo "  Done. Service enabled (starts on boot)."

# ----- Complete -----
echo ""
echo "============================================"
echo "  Setup Complete"
echo "============================================"
echo ""
echo "Run manually:"
echo "  cd $PROJECT_DIR"
echo "  source venv/bin/activate"
echo "  python -m rpi_assistant.main"
echo ""
echo "Run as service:"
echo "  sudo systemctl start vision-assistant"
echo "  sudo systemctl status vision-assistant"
echo "  journalctl -u vision-assistant -f"
echo ""
if [ -z "$GEMINI_KEY" ] || [ "$GEMINI_KEY" = "your-api-key-here" ]; then
    echo "NOTE: Gemini API key not set."
    echo "  Edit .env and add your key, then re-run this script"
    echo "  or update the systemd service manually."
    echo ""
fi
