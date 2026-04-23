"""
config.py — Central configuration for AC Anomaly Detector
Adjust CAMERA_INDEX and ANOMALY_CLASSES to match your setup
"""

# ── Camera ──────────────────────────────────────────────────────────────────
# 0 = first USB cam. Try 1 if 0 doesn't work.
# For CSI camera on Jetson use the gstreamer string instead (see README).
CAMERA_INDEX = 0

FRAME_WIDTH  = 1280
FRAME_HEIGHT = 720

# ── YOLO Model ───────────────────────────────────────────────────────────────
# yolov8n.pt = nano (fastest on Jetson, auto-downloaded on first run)
# yolov8s.pt = small (more accurate but slower)
YOLO_MODEL   = "yolov8n.pt"
YOLO_CONF    = 0.40   # Minimum confidence threshold (0.0–1.0)
YOLO_IOU     = 0.45   # NMS IOU threshold

# ── Anomaly Classification ───────────────────────────────────────────────────
# Objects in this list are treated as ANOMALIES (red / high-threat)
# Everything else is treated as NORMAL (blue / neutral)
# Full COCO class list: https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml
ANOMALY_CLASSES = [
    "knife",
    "scissors",
    "gun",            # not in standard COCO but kept for custom models
    "cell phone",     # suspicious loitering signal
  
]

# Objects explicitly marked as neutral (safe)
NEUTRAL_CLASSES = [
    "person",
    "chair",
    "laptop",
    "keyboard",
    "mouse",
    "book",
    "cup",
    "desk",
    "tv",
    "monitor",
]

# ── AC HUD Appearance ─────────────────────────────────────────────────────────
# Eagle Vision color palette (BGR for OpenCV)
COLOR_ANOMALY   = (0,   50, 220)   # Red  — hostile
COLOR_NEUTRAL   = (200, 130, 40)   # Gold — civilian
COLOR_TRACKED   = (200, 160, 30)   # Amber— tracked
COLOR_GOLD      = (60,  180, 210)  # HUD gold
COLOR_HUD_TEXT  = (180, 220, 240)  # Light cream
COLOR_DARK      = (10,   8,   5)   # Near black
COLOR_EAGLE_BG  = (30,  20,  10)   # Eagle vision tint

# Alert thresholds
ANOMALY_ALERT_COUNT = 1   # How many anomalies trigger THREAT DETECTED banner
HIGH_THREAT_CONF    = 0.75  # Above this confidence = HIGH threat
