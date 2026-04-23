"""
detector.py — YOLOv8 object detection + anomaly classification
"""

import time
from ultralytics import YOLO
from config import (
    YOLO_MODEL, YOLO_CONF, YOLO_IOU,
    ANOMALY_CLASSES, NEUTRAL_CLASSES
)


class Detection:
    """Holds one detected object with its classification."""
    def __init__(self, label, confidence, bbox, is_anomaly, threat_level):
        self.label       = label          # class name string
        self.confidence  = confidence     # float 0–1
        self.bbox        = bbox           # (x1, y1, x2, y2) ints
        self.is_anomaly  = is_anomaly     # bool
        self.threat_level = threat_level  # "HIGH" | "MEDIUM" | "LOW" | "CLEAR"
        self.timestamp   = time.time()

    @property
    def cx(self):
        return (self.bbox[0] + self.bbox[2]) // 2

    @property
    def cy(self):
        return (self.bbox[1] + self.bbox[3]) // 2


class AnomalyDetector:
    def __init__(self):
        print(f"[DETECTOR] Loading YOLO model: {YOLO_MODEL}")
        self.model = YOLO(YOLO_MODEL)
        # Warm-up pass to pre-load weights
        import numpy as np
        dummy = np.zeros((640, 640, 3), dtype="uint8")
        self.model(dummy, verbose=False)
        print("[DETECTOR] YOLO ready.")

    def detect(self, frame):
        """Run YOLO inference. Returns raw results."""
        results = self.model(
            frame,
            conf=YOLO_CONF,
            iou=YOLO_IOU,
            verbose=False
        )
        return results

    def classify(self, results):
        """
        Convert raw YOLO results into a list of Detection objects
        with anomaly classification.
        """
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                label = result.names[int(box.cls[0])]
                conf  = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                is_anomaly   = label.lower() in [a.lower() for a in ANOMALY_CLASSES]
                threat_level = self._threat_level(label, conf, is_anomaly)

                detections.append(Detection(
                    label=label,
                    confidence=conf,
                    bbox=(x1, y1, x2, y2),
                    is_anomaly=is_anomaly,
                    threat_level=threat_level
                ))

        # Sort: anomalies first, then by confidence descending
        detections.sort(key=lambda d: (not d.is_anomaly, -d.confidence))
        return detections

    def _threat_level(self, label, conf, is_anomaly):
        if not is_anomaly:
            return "CLEAR"
        if conf >= 0.75:
            return "HIGH"
        if conf >= 0.55:
            return "MEDIUM"
        return "LOW"
