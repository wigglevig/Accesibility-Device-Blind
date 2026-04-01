"""
Central configuration for the Smart Vision Assistant.
Edit this file to customize behavior for your hardware setup.
"""
import os
from dotenv import load_dotenv

# Load environment variables from .env file (project root)
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

# ============================================================
# HARDWARE CONFIGURATION
# ============================================================

# Camera
CAMERA_INDEX = 0                    # USB camera index (usually 0)
CAMERA_WIDTH = 640                  # Frame width (lower = faster on RPi)
CAMERA_HEIGHT = 480                 # Frame height

# GPIO Pin Assignments (BCM numbering)
# Set to None to disable a component you don't have wired
BUTTON_DESCRIBE_PIN = None          # Button 1: "Describe scene" (using keyboard instead)
BUTTON_ASK_PIN = None               # Button 2: "Ask AI a question" (using keyboard)
BUTTON_SUMMARY_PIN = None           # Button 3: "What objects are around me?" (using keyboard)

ULTRASONIC_TRIGGER_PIN = None       # HC-SR04 Trigger (set to 23 when wired)
ULTRASONIC_ECHO_PIN = None          # HC-SR04 Echo (set to 24 when wired)
VIBRATION_MOTOR_PIN = None          # Vibration motor (set to 25 when wired)

# ============================================================
# AUDIO CONFIGURATION
# ============================================================

TTS_RATE = 160                      # Words per minute (default 150-200)
TTS_VOLUME = 1.0                    # Volume 0.0 to 1.0
VOICE_LISTEN_TIMEOUT = 5            # Seconds to wait for voice input
VOICE_PHRASE_LIMIT = 8              # Max seconds for a single phrase

# ============================================================
# DETECTION CONFIGURATION
# ============================================================

YOLO_MODEL = "yolov8n.pt"           # YOLOv8 nano (lightest, best for RPi)
YOLO_CONFIDENCE = 0.45              # Minimum confidence threshold
YOLO_FRAME_SKIP = 3                # Process every Nth frame in continuous mode

# Spatial zones for describing object positions
# Frame is divided into 3x3 grid (left/center/right × top/middle/bottom)
POSITION_THRESHOLD_X = 0.33         # Left/center/right boundary (fraction of width)
POSITION_THRESHOLD_Y = 0.33         # Top/middle/bottom boundary (fraction of height)

# ============================================================
# DISTANCE / SAFETY CONFIGURATION
# ============================================================

# Ultrasonic sensor thresholds (centimeters)
DISTANCE_DANGER = 30                # 🔴 Urgent alert + vibration
DISTANCE_CAUTION = 100              # 🟡 Spoken warning
DISTANCE_SAFE = 200                 # 🟢 No alert

ALERT_COOLDOWN = 3.0                # Seconds between repeated alerts
SAFETY_CHECK_INTERVAL = 0.3         # Seconds between distance checks

# ============================================================
# AI ASSISTANT CONFIGURATION (Gemini)
# ============================================================

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-2.0-flash"   # Fast + vision capable
GEMINI_MAX_TOKENS = 200             # Keep responses concise for TTS

# System prompts for different modes
SCENE_DESCRIPTION_PROMPT = (
    "You are a vision assistant for a blind person. "
    "Describe what you see in this image concisely in 2-3 sentences. "
    "Mention: important objects, their approximate positions (left/center/right), "
    "any text visible, potential hazards (stairs, vehicles, wet floor), "
    "and the general environment (indoor/outdoor, lighting). "
    "Speak naturally as if talking to the person. Do NOT use bullet points or markdown."
)

QA_SYSTEM_PROMPT = (
    "You are a helpful vision assistant for a blind person. "
    "The user is asking about what's in front of them based on the camera image. "
    "Answer concisely in 1-2 sentences. Be specific and helpful. "
    "If you can't determine the answer from the image, say so honestly."
)

CONVERSATION_MEMORY_SIZE = 5        # Number of past exchanges to remember

# ============================================================
# PLATFORM DETECTION
# ============================================================

def is_raspberry_pi():
    """Detect if running on Raspberry Pi."""
    try:
        with open("/proc/device-tree/model", "r") as f:
            return "raspberry pi" in f.read().lower()
    except (FileNotFoundError, PermissionError):
        return False

RUNNING_ON_RPI = is_raspberry_pi()

# ============================================================
# INPUT MODE
# ============================================================

# Set to True to use keyboard (type 1/2/3/q) instead of GPIO buttons.
# Useful when running on RPi without physical buttons wired up,
# e.g. with a USB keyboard + monitor + earphones setup.
# Set to False when you have physical GPIO buttons connected.
USE_KEYBOARD_INPUT = True

# When not on RPi, always force keyboard mode
DEV_MODE = not RUNNING_ON_RPI
if DEV_MODE:
    USE_KEYBOARD_INPUT = True
    print("[CONFIG] Running in DEVELOPMENT mode (keyboard controls)")
elif USE_KEYBOARD_INPUT:
    print("[CONFIG] Running on Raspberry Pi (keyboard controls)")
else:
    print("[CONFIG] Running on Raspberry Pi (GPIO controls)")
