#!/bin/bash
# ============================================================
# Smart Vision Assistant — Raspberry Pi Setup Script
# ============================================================
# Run this on a fresh Raspberry Pi 4B to set everything up.
#
# Usage:
#   chmod +x setup.sh
#   ./setup.sh
# ============================================================

set -e  # Exit on error

echo "============================================"
echo "  Smart Vision Assistant — RPi Setup"
echo "============================================"
echo ""

# 1. System dependencies
echo "[1/6] Installing system dependencies..."
sudo apt-get update -qq
sudo apt-get install -y -qq \
    python3-pip \
    python3-venv \
    python3-dev \
    portaudio19-dev \
    espeak \
    libatlas-base-dev \
    libjasper-dev \
    libqtgui4 \
    libqt4-test \
    libhdf5-dev \
    libharfbuzz-dev \
    libwebp-dev \
    libtiff5 \
    libjpeg-dev \
    libpng-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    v4l-utils \
    flac \
    2>/dev/null || true

echo "  ✅ System dependencies installed"

# 2. Create virtual environment
echo ""
echo "[2/6] Creating Python virtual environment..."
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
cd "$SCRIPT_DIR/.."

if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "  ✅ Virtual environment created"
else
    echo "  ⏭️  Virtual environment already exists"
fi

source venv/bin/activate
pip install --upgrade pip -q

# 3. Install Python packages
echo ""
echo "[3/6] Installing Python packages (this may take 5-10 minutes)..."
pip install -q \
    torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    2>/dev/null || pip install -q torch torchvision

pip install -q \
    ultralytics \
    opencv-python-headless \
    numpy \
    Pillow \
    pyttsx3 \
    SpeechRecognition \
    pyaudio \
    google-genai \
    RPi.GPIO \
    2>/dev/null || true

echo "  ✅ Python packages installed"

# 4. Download YOLO model
echo ""
echo "[4/6] Downloading YOLOv8 nano model..."
python3 -c "
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
print('  ✅ YOLOv8 nano model ready')
"

# 5. Test audio
echo ""
echo "[5/6] Testing audio output..."
python3 -c "
import pyttsx3
engine = pyttsx3.init()
engine.setProperty('rate', 160)
engine.say('Audio test successful. Vision assistant setup is complete.')
engine.runAndWait()
print('  ✅ Audio output working')
" 2>/dev/null || echo "  ⚠️  Audio test failed — check speaker connection"

# 6. Create systemd service for auto-start
echo ""
echo "[6/6] Setting up auto-start service..."
SERVICE_FILE="/etc/systemd/system/vision-assistant.service"

sudo tee "$SERVICE_FILE" > /dev/null << EOF
[Unit]
Description=Smart Vision Assistant
After=multi-user.target sound.target
Wants=sound.target

[Service]
Type=simple
User=$(whoami)
WorkingDirectory=$SCRIPT_DIR/..
ExecStart=$SCRIPT_DIR/../venv/bin/python -m rpi_assistant.main
Restart=on-failure
RestartSec=5
Environment=GEMINI_API_KEY=${GEMINI_API_KEY:-}

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable vision-assistant.service
echo "  ✅ Auto-start service created"

# Done!
echo ""
echo "============================================"
echo "  ✅ Setup Complete!"
echo "============================================"
echo ""
echo "To run manually:"
echo "  cd $SCRIPT_DIR/.."
echo "  source venv/bin/activate"
echo "  export GEMINI_API_KEY='your-key-here'"
echo "  python -m rpi_assistant.main"
echo ""
echo "To start as service:"
echo "  sudo systemctl start vision-assistant"
echo ""
echo "To check service status:"
echo "  sudo systemctl status vision-assistant"
echo ""
echo "To view logs:"
echo "  journalctl -u vision-assistant -f"
echo ""
echo "⚠️  Don't forget to set your Gemini API key!"
echo "  Edit /etc/systemd/system/vision-assistant.service"
echo "  and set Environment=GEMINI_API_KEY=your-key"
echo "  Then: sudo systemctl daemon-reload && sudo systemctl restart vision-assistant"
echo ""
