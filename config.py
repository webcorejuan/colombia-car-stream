import os
from dotenv import load_dotenv

load_dotenv()

# ==========================================
# 1. CAMERA & STREAM SETTINGS (Logitech Brio 4K)
# ==========================================
# On Windows, 0 is usually the built-in/first webcam, 1 is the second USB camera.
# You can also put a video file path here (e.g., "test_street.mp4") to test before mounting the camera!
CAMERA_SOURCE = os.getenv("CAMERA_SOURCE", "0")
try:
    CAMERA_SOURCE = int(CAMERA_SOURCE)
except ValueError:
    pass  # Keep as string if it's a video path or RTSP URL

# We capture at 1920x1080 @ 30fps on the Logitech Brio for the best balance of
# sharp car crops + cool 24/7 thermal performance on the USB bus and RTX 4060.
FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080
TARGET_FPS = 30

# ==========================================
# 2. YOLO TRACKING & TRIPWIRE SETTINGS
# ==========================================
# YOLOv11s (small) runs at ~120 FPS on RTX 4060 in FP16
YOLO_MODEL = "yolo11s.pt"
CONFIDENCE_THRESHOLD = 0.40

# COCO classes: 2 = car, 3 = motorcycle, 5 = bus, 7 = truck
TRACKED_CLASSES = [2, 3, 5, 7]

# Virtual Tripwire Line Position (0.0 = top of frame, 1.0 = bottom of frame)
# When a vehicle's center crosses this horizontal line, we crop it and classify the brand.
COUNTING_LINE_Y_RATIO = 0.62

# Minimum pixel width/height of a cropped car to send to the Vision model (avoids tiny blurry blobs)
MIN_CROP_SIZE = 90

# ==========================================
# 3. AI BRAND CLASSIFIER (100% LOCAL OLLAMA + OPTIONAL API)
# ==========================================
# Mode: "local" (100% free via Ollama on RTX 4060) or "openrouter" (cheap cloud fallback)
CLASSIFIER_MODE = os.getenv("CLASSIFIER_MODE", "local")

# Ollama local settings (qwen2.5vl:3b fits alongside YOLOv11s in 8GB VRAM)
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5vl:3b")

# Optional fallback API (e.g. OpenRouter Qwen-VL-Plus or Gemini Flash-Lite)
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "qwen/qwen-2.5-vl-7b-instruct")

# ==========================================
# 4. COLOMBIAN CAR BRANDS & SPECIAL ALERTS
# ==========================================
KNOWN_BRANDS = [
    "Renault",
    "Chevrolet",
    "Mazda",
    "Kia",
    "Toyota",
    "Suzuki",
    "Nissan",
    "Hyundai",
    "Volkswagen",
    "Ford",
    "BYD",
    "Chery",
    "JAC",
    "Bajaj",
    "Yamaha",
    "Honda",
    "AKT",
    "Victory",
    "Other",
]

# State persistence file (so if you restart the PC, today's counters aren't lost)
STATS_FILE = "daily_stats.json"
