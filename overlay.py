"""
overlay.py — Assassin's Creed themed HUD overlay
Draws Eagle Vision bounding boxes, threat panel, minimap, event log.
"""

import cv2
import numpy as np
import time
import math
from collections import deque
from config import (
    COLOR_ANOMALY, COLOR_NEUTRAL, COLOR_GOLD, COLOR_HUD_TEXT,
    COLOR_DARK, COLOR_EAGLE_BG, COLOR_TRACKED, COLOR_HUD_TEXT,
    ANOMALY_ALERT_COUNT, HIGH_THREAT_CONF, FRAME_WIDTH, FRAME_HEIGHT
)

# ── Font shortcuts ────────────────────────────────────────────────────────────
FONT       = cv2.FONT_HERSHEY_DUPLEX
FONT_SMALL = cv2.FONT_HERSHEY_SIMPLEX
FONT_MONO  = cv2.FONT_HERSHEY_PLAIN


def draw_text(img, text, pos, font=FONT_SMALL, scale=0.45, color=None,
              thickness=1, shadow=True):
    color = color or COLOR_HUD_TEXT
    x, y = pos
    if shadow:
        cv2.putText(img, text, (x+1, y+1), font, scale, (0,0,0), thickness+1, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


def draw_corner_bracket(img, x1, y1, x2, y2, color, size=14, thick=2):
    """Draw AC-style corner brackets around a bounding box."""
    pts = [
        # top-left
        [(x1, y1+size), (x1, y1), (x1+size, y1)],
        # top-right
        [(x2-size, y1), (x2, y1), (x2, y1+size)],
        # bottom-left
        [(x1, y2-size), (x1, y2), (x1+size, y2)],
        # bottom-right
        [(x2-size, y2), (x2, y2), (x2, y2-size)],
    ]
    for bracket in pts:
        for i in range(len(bracket) - 1):
            cv2.line(img, bracket[i], bracket[i+1], color, thick, cv2.LINE_AA)


def draw_panel_bg(img, x, y, w, h, alpha=0.72):
    """Semi-transparent dark panel."""
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x+w, y+h), (10, 8, 5), -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    # Gold border
    cv2.rectangle(img, (x, y), (x+w, y+h), COLOR_GOLD, 1, cv2.LINE_AA)
    # Top accent line
    cv2.line(img, (x, y), (x+w, y), COLOR_GOLD, 2, cv2.LINE_AA)


def draw_gold_line(img, x1, y1, x2, y2):
    cv2.line(img, (x1, y1), (x2, y2), COLOR_GOLD, 1, cv2.LINE_AA)


class ACOverlay:
    def __init__(self, width, height):
        self.W = width
        self.H = height
        self.event_log = deque(maxlen=8)
        self.alert_flash_time = 0
        self.session_start = time.time()
        self.total_anomalies_seen = 0
        self.prev_labels = set()

    def reset_log(self):
        self.event_log.clear()

    def _log_event(self, text, level="INFO"):
        ts = time.strftime("%H:%M:%S")
        self.event_log.appendleft(f"[{ts}] [{level}] {text}")

    # ── Main draw entry point ─────────────────────────────────────────────────
    def draw(self, frame, detections, fps, eagle_vision_on):
        output = frame.copy()

        if eagle_vision_on:
            output = self._apply_eagle_vision_tint(output)

        # Detect new anomaly events for logging
        current_labels = {d.label for d in detections if d.is_anomaly}
        new_labels = current_labels - self.prev_labels
        for label in new_labels:
            self.total_anomalies_seen += 1
            self._log_event(f"ANOMALY DETECTED: {label.upper()}", "ALERT")
            self.alert_flash_time = time.time()
        self.prev_labels = current_labels

        # Draw bounding boxes
        for det in detections:
            self._draw_detection(output, det)

        # Panels
        self._draw_top_bar(output, fps, detections)
        self._draw_left_panel(output, detections)
        self._draw_right_panel(output, detections)
        self._draw_bottom_bar(output)
        self._draw_crosshair(output)
        self._draw_corner_decor(output)

        # Alert banner if anomaly present
        anomaly_count = sum(1 for d in detections if d.is_anomaly)
        if anomaly_count >= ANOMALY_ALERT_COUNT:
            self._draw_alert_banner(output, anomaly_count)

        return output

    # ── Eagle Vision color tint ───────────────────────────────────────────────
    def _apply_eagle_vision_tint(self, frame):
        """Apply a dark blue-gold tint to simulate Eagle Vision."""
        tinted = frame.copy()
        # Desaturate slightly
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        # Mix original with dark blue tint
        tint = np.full_like(frame, (25, 15, 5), dtype=np.uint8)
        tinted = cv2.addWeighted(gray_bgr, 0.55, tint, 0.15, 0)
        tinted = cv2.addWeighted(tinted, 0.85, frame, 0.15, 0)
        return tinted

    # ── Single detection box ──────────────────────────────────────────────────
    def _draw_detection(self, img, det):
        x1, y1, x2, y2 = det.bbox
        color = COLOR_ANOMALY if det.is_anomaly else COLOR_NEUTRAL

        # Filled semi-transparent box
        overlay = img.copy()
        alpha = 0.12 if det.is_anomaly else 0.06
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        cv2.addWeighted(overlay, alpha, img, 1-alpha, 0, img)

        # Corner brackets (AC style)
        draw_corner_bracket(img, x1, y1, x2, y2, color, size=16, thick=2)

        # Label tag
        threat_map = {"HIGH": "!! HIGH", "MEDIUM": "! MED", "LOW": "LOW", "CLEAR": "CLEAR"}
        tag = f"{det.label.upper()}  {det.confidence:.0%}  {threat_map.get(det.threat_level,'')}"
        tag_w = cv2.getTextSize(tag, FONT_SMALL, 0.42, 1)[0][0]

        label_y = max(y1 - 4, 20)
        label_x = x1

        # Tag background
        cv2.rectangle(img, (label_x - 2, label_y - 13),
                      (label_x + tag_w + 6, label_y + 2), color, -1)
        draw_text(img, tag, (label_x + 2, label_y - 2),
                  scale=0.42, color=(255, 255, 255), shadow=False)

        # Pulse dot on anomaly
        if det.is_anomaly:
            pulse = abs(math.sin(time.time() * 4)) * 5
            cx, cy = det.cx, det.cy
            cv2.circle(img, (cx, cy), int(4 + pulse), color, 1, cv2.LINE_AA)
            cv2.circle(img, (cx, cy), 3, color, -1, cv2.LINE_AA)

    # ── Top bar ───────────────────────────────────────────────────────────────
    def _draw_top_bar(self, img, fps, detections):
        H, W = img.shape[:2]
        draw_panel_bg(img, 0, 0, W, 38, alpha=0.80)

        # Left: AC logo text
        draw_text(img, "ANIMUS // EAGLE VISION", (12, 24),
                  font=FONT, scale=0.55, color=COLOR_GOLD, thickness=1)

        # Center: sync status
        elapsed = int(time.time() - self.session_start)
        sync_text = f"SYNC: {elapsed:04d}s    FPS: {fps:4.1f}"
        tw = cv2.getTextSize(sync_text, FONT_SMALL, 0.45, 1)[0][0]
        draw_text(img, sync_text, (W//2 - tw//2, 24),
                  scale=0.45, color=COLOR_HUD_TEXT)

        # Right: anomaly count
        n_anom = sum(1 for d in detections if d.is_anomaly)
        n_obj  = len(detections)
        status = f"OBJECTS: {n_obj:02d}   THREATS: {n_anom:02d}"
        tw2 = cv2.getTextSize(status, FONT_SMALL, 0.45, 1)[0][0]
        col = COLOR_ANOMALY if n_anom > 0 else COLOR_GOLD
        draw_text(img, status, (W - tw2 - 12, 24),
                  scale=0.45, color=col)

        # Separator line
        draw_gold_line(img, 0, 38, W, 38)

    # ── Left panel — threat list ──────────────────────────────────────────────
    def _draw_left_panel(self, img, detections):
        px, py, pw, ph = 10, 48, 230, 30 + min(len(detections), 6) * 22 + 40
        draw_panel_bg(img, px, py, pw, ph)

        draw_text(img, "// TARGET ANALYSIS", (px+8, py+16),
                  scale=0.40, color=COLOR_GOLD)
        draw_gold_line(img, px+4, py+20, px+pw-4, py+20)

        y_cursor = py + 34
        shown = detections[:6]

        for det in shown:
            color = COLOR_ANOMALY if det.is_anomaly else COLOR_NEUTRAL
            # Dot indicator
            cv2.circle(img, (px+12, y_cursor-4), 4, color, -1, cv2.LINE_AA)

            label_str = f"{det.label[:14]:<14}"
            conf_str  = f"{det.confidence:.0%}"
            draw_text(img, label_str, (px+22, y_cursor),
                      scale=0.40, color=color)
            draw_text(img, conf_str, (px+pw-38, y_cursor),
                      scale=0.40, color=COLOR_HUD_TEXT)
            y_cursor += 22

        if not detections:
            draw_text(img, "NO TARGETS IN RANGE", (px+14, y_cursor),
                      scale=0.38, color=(80, 80, 80))

        # Total at bottom
        draw_gold_line(img, px+4, py+ph-18, px+pw-4, py+ph-18)
        draw_text(img, f"SESSION ANOMALIES: {self.total_anomalies_seen:03d}",
                  (px+8, py+ph-5), scale=0.38, color=COLOR_GOLD)

    # ── Right panel — event log ───────────────────────────────────────────────
    def _draw_right_panel(self, img, detections):
        H, W = img.shape[:2]
        pw, ph = 270, 180
        px = W - pw - 10
        py = 48
        draw_panel_bg(img, px, py, pw, ph)

        draw_text(img, "// EVENT LOG", (px+8, py+16),
                  scale=0.40, color=COLOR_GOLD)
        draw_gold_line(img, px+4, py+20, px+pw-4, py+20)

        y_log = py + 34
        for entry in list(self.event_log)[:6]:
            col = COLOR_ANOMALY if "ALERT" in entry else (100, 160, 180)
            # Truncate long entries
            display = entry[:40] if len(entry) > 40 else entry
            draw_text(img, display, (px+8, y_log),
                      scale=0.33, color=col)
            y_log += 22

        if not self.event_log:
            draw_text(img, "MONITORING...", (px+8, y_log),
                      scale=0.38, color=(60, 60, 60))

    # ── Bottom bar ────────────────────────────────────────────────────────────
    def _draw_bottom_bar(self, img):
        H, W = img.shape[:2]
        draw_panel_bg(img, 0, H-28, W, 28, alpha=0.80)
        draw_gold_line(img, 0, H-28, W, H-28)

        controls = "[Q] QUIT    [E] EAGLE VISION    [S] SCREENSHOT    [R] RESET LOG"
        tw = cv2.getTextSize(controls, FONT_SMALL, 0.38, 1)[0][0]
        draw_text(img, controls, (W//2 - tw//2, H-10),
                  scale=0.38, color=(100, 130, 150))

        # Left motto
        draw_text(img, "NOTHING IS TRUE, EVERYTHING IS PERMITTED",
                  (10, H-10), scale=0.34, color=COLOR_GOLD)

    # ── Center crosshair ─────────────────────────────────────────────────────
    def _draw_crosshair(self, img):
        H, W = img.shape[:2]
        cx, cy = W//2, H//2
        size = 18
        gap  = 6
        color = COLOR_GOLD

        # Horizontal
        cv2.line(img, (cx - size, cy), (cx - gap, cy), color, 1, cv2.LINE_AA)
        cv2.line(img, (cx + gap, cy), (cx + size, cy), color, 1, cv2.LINE_AA)
        # Vertical
        cv2.line(img, (cx, cy - size), (cx, cy - gap), color, 1, cv2.LINE_AA)
        cv2.line(img, (cx, cy + gap), (cx, cy + size), color, 1, cv2.LINE_AA)
        # Center dot
        cv2.circle(img, (cx, cy), 2, color, -1, cv2.LINE_AA)

    # ── Corner decorations ────────────────────────────────────────────────────
    def _draw_corner_decor(self, img):
        H, W = img.shape[:2]
        size = 24
        thick = 2
        c = COLOR_GOLD

        # Bottom-left
        cv2.line(img, (0, H-28), (size, H-28), c, thick, cv2.LINE_AA)
        cv2.line(img, (0, H-28), (0, H-28-size), c, thick, cv2.LINE_AA)

        # Bottom-right
        cv2.line(img, (W, H-28), (W-size, H-28), c, thick, cv2.LINE_AA)
        cv2.line(img, (W, H-28), (W, H-28-size), c, thick, cv2.LINE_AA)

    # ── Alert banner ──────────────────────────────────────────────────────────
    def _draw_alert_banner(self, img, count):
        H, W = img.shape[:2]

        # Flashing alpha
        flash = abs(math.sin(time.time() * 3.5))
        alpha = 0.55 + 0.30 * flash

        overlay = img.copy()
        bh = 44
        by = H//2 - bh//2
        cv2.rectangle(overlay, (0, by), (W, by+bh), (0, 0, 139), -1)
        cv2.addWeighted(overlay, alpha, img, 1-alpha, 0, img)

        # Top/bottom border lines
        cv2.line(img, (0, by),     (W, by),     COLOR_ANOMALY, 2, cv2.LINE_AA)
        cv2.line(img, (0, by+bh),  (W, by+bh),  COLOR_ANOMALY, 2, cv2.LINE_AA)

        msg = f"!! ANOMALY DETECTED — {count} THREAT{'S' if count>1 else ''} IN SECTOR !!"
        tw = cv2.getTextSize(msg, FONT, 0.75, 2)[0][0]

        # Shadow
        cv2.putText(img, msg, (W//2 - tw//2 + 2, by+30), FONT,
                    0.75, (0, 0, 0), 3, cv2.LINE_AA)
        # Text
        col_r = int(180 + 75 * flash)
        cv2.putText(img, msg, (W//2 - tw//2, by+28), FONT,
                    0.75, (0, 0, col_r), 2, cv2.LINE_AA)
