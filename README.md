# 🦅 AC Anomaly Detector — Lab #5

> **Real-time anomaly detection with an Assassin's Creed Eagle Vision HUD**  
> Running on **Jetson Nano 4GB** · YOLOv8n · OpenCV · Python

---

## Overview

This project implements a real-time object-based anomaly detector that runs entirely locally on a Jetson Nano 4GB with a connected USB or CSI camera. The system uses **YOLOv8 nano** to detect objects in each frame and classifies them as **threats (anomalies)** or **neutral** based on a configurable class list.

The UI is styled after **Assassin's Creed's Eagle Vision** — a dark, high-contrast overlay with gold corner brackets, threat panels, a live event log, and a flashing red alert banner when an anomaly is detected.

---

## Features

- **Real-time object detection** via YOLOv8n (no custom training needed)
- **Anomaly classification** — configurable threat class list in `config.py`
- **AC Eagle Vision HUD**:
  - Gold corner bracket bounding boxes
  - Color-coded threat levels (Red = anomaly, Gold = neutral)
  - Live event log (timestamped, right panel)
  - Target analysis panel (left panel) with confidence scores
  - Flashing red THREAT DETECTED banner
  - Center crosshair
  - FPS counter, session timer, object count
- **Screenshot capture** with `S` key
- **Eagle Vision toggle** with `E` key

---

## Hardware

| Component | Details |
|---|---|
| Board | NVIDIA Jetson Nano 4GB |
| Camera | USB webcam (or CSI camera) |
| OS | Ubuntu 18.04 / JetPack 4.x |
| Python | 3.8+ |

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/YOUR_USERNAME/ac-anomaly-detector.git
cd ac-anomaly-detector
```

### 2. Install dependencies

```bash
pip3 install -r requirements.txt
```

> `ultralytics` will auto-download `yolov8n.pt` (~6 MB) on first run.

### 3. (Optional) CSI Camera on Jetson

If using a CSI camera instead of USB, open `config.py` and change:

```python
CAMERA_INDEX = "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=1280, height=720, framerate=30/1 ! nvvidconv ! video/x-raw, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink"
```

### 4. Run

```bash
python3 main.py
```

---

## Controls

| Key | Action |
|---|---|
| `Q` | Quit |
| `E` | Toggle Eagle Vision tint |
| `S` | Save screenshot |
| `R` | Clear event log |

---

## Project Structure

```
ac_anomaly_detector/
├── main.py          # Entry point — camera loop & key controls
├── detector.py      # YOLOv8 wrapper + anomaly classification logic
├── overlay.py       # All AC HUD drawing (OpenCV)
├── config.py        # Camera, model, color, and class settings
├── requirements.txt # Python dependencies
└── README.md        # This file
```

---

## How Anomaly Detection Works

1. Each frame from the camera is passed to **YOLOv8n**
2. All detected objects are compared against the `ANOMALY_CLASSES` list in `config.py`
3. Matching objects are flagged as **anomalies** and assigned a threat level:
   - `HIGH` → confidence ≥ 75%
   - `MEDIUM` → confidence 55–74%
   - `LOW` → confidence < 55%
4. When ≥ 1 anomaly is detected, a **THREAT DETECTED** banner flashes on screen
5. All events are timestamped and logged in the right HUD panel

### Default Anomaly Classes

```
knife, scissors, cell phone, backpack,
handbag, suitcase, baseball bat, bottle
```

These are all detectable by standard YOLOv8 trained on COCO. You can add or remove any class from `config.py`.

---

## Results

| Metric | Value |
|---|---|
| Model | YOLOv8n (nano) |
| Inference speed | ~15–25 FPS on Jetson Nano |
| Confidence threshold | 0.40 |
| Detection classes | 80 (COCO) |
| Anomaly trigger classes | 8 (configurable) |

### Screenshots

> *(Add your screenshots here — use `S` key to capture while running)*

---

## UI Theme: Assassin's Creed Eagle Vision

The HUD is inspired by the **Eagle Vision** ability from the Assassin's Creed franchise — enemies highlighted in red, civilians in gold, with a dark desaturated environment overlay. All UI elements (panels, brackets, lines) follow the AC visual language.

---

## License

MIT License — free to use, modify, and share.
