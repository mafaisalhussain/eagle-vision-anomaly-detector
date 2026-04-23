"""
Assassin's Creed Anomaly Detector
Lab #5 — Object Detection with AC-themed HUD
Runs on Jetson Nano 4GB with USB/CSI camera
"""

import cv2
import time
import sys
from detector import AnomalyDetector
from overlay import ACOverlay
from config import CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT, ANOMALY_CLASSES

def main():
    print("\n[ANIMUS] Synchronizing with the Grid...")
    print("[ANIMUS] Loading Eagle Vision system...\n")

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        print("[ERROR] Camera not found. Check CAMERA_INDEX in config.py")
        sys.exit(1)

    detector = AnomalyDetector()
    overlay  = ACOverlay(FRAME_WIDTH, FRAME_HEIGHT)

    print("[ANIMUS] Eagle Vision ONLINE")
    print("[CONTROLS] Q = Quit | E = Toggle Eagle Vision | S = Screenshot | R = Reset log\n")

    eagle_vision_on = True
    frame_count = 0
    fps_time = time.time()
    fps = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to read from camera.")
            break

        frame_count += 1

        # FPS calculation every 30 frames
        if frame_count % 30 == 0:
            now = time.time()
            fps = 30.0 / (now - fps_time)
            fps_time = now

        # Run YOLO detection
        detections = detector.detect(frame)

        # Classify each detection as anomaly or normal
        classified = detector.classify(detections)

        # Draw AC-themed overlay
        output = overlay.draw(frame, classified, fps, eagle_vision_on)

        cv2.imshow("ANIMUS // EAGLE VISION ANOMALY DETECTOR", output)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q') or key == ord('Q'):
            print("\n[ANIMUS] Desynchronizing...")
            break

        elif key == ord('e') or key == ord('E'):
            eagle_vision_on = not eagle_vision_on
            state = "ON" if eagle_vision_on else "OFF"
            print(f"[EAGLE VISION] {state}")

        elif key == ord('s') or key == ord('S'):
            ts = int(time.time())
            fname = f"screenshot_{ts}.png"
            cv2.imwrite(fname, output)
            print(f"[SCREENSHOT] Saved → {fname}")

        elif key == ord('r') or key == ord('R'):
            overlay.reset_log()
            print("[LOG] Event log cleared.")

    cap.release()
    cv2.destroyAllWindows()
    print("[ANIMUS] Session ended. Nothing is true, everything is permitted.\n")

if __name__ == "__main__":
    main()
