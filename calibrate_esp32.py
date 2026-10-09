"""
SPRS ESP32-CAM Dedicated Calibration Utility (calibrate_esp32.py)

Exclusive Reference Object: Earbuds Case (Length: 60.2 mm / 6.02 cm)
Camera Mounting Height: 20.0 cm above packing surface
"""

import sys
import os
import json
import time
import cv2
from sprs.calibration import CalibrationManager, EARBUDS_CASE_LENGTH_CM
from sprs.camera_stream import CameraStream

ESP32_CONFIG_FILE = "esp32_camera_config.json"


def load_esp32_url() -> str:
    if os.path.exists(ESP32_CONFIG_FILE):
        try:
            with open(ESP32_CONFIG_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("esp32_url", "http://192.168.1.100:81/stream")
        except Exception:
            pass
    return "http://192.168.1.100:81/stream"


def main():
    print("\n=======================================================")
    print("  SPRS ESP32-CAM DEDICATED CALIBRATION UTILITY")
    print(f"  Reference Object: Earbuds Case (Length: {EARBUDS_CASE_LENGTH_CM} cm / 60.2 mm)")
    print("  Camera Mounting Height: 20.0 cm above packing surface")
    print("=======================================================\n")
    
    calib_mgr = CalibrationManager()
    print(f"Current Scale: {calib_mgr.pixels_per_cm:.2f} px/cm")

    esp32_url = load_esp32_url()
    input_url = input(f"\nEnter ESP32-CAM stream URL [Default: {esp32_url}]: ").strip()
    if input_url:
        if not input_url.startswith("http"):
            input_url = f"http://{input_url}"
        esp32_url = input_url

    cam_stream = CameraStream(esp32_url)
    if not cam_stream.start():
        print(f"[Error] Cannot open ESP32-CAM stream at '{esp32_url}'.")
        return

    print(f"\nLive ESP32-CAM stream connected: '{esp32_url}'.")
    print("1. Place your earbuds case flat under your ESP32-CAM (Height: 20 cm).")
    print("2. Click START and END points along the 6.02 cm edge of your earbuds case.")
    print("3. Press ENTER or 'C' to save calibration scale.\n")

    new_px_cm = calib_mgr.interactive_calibrate_live(cam_stream)
    cam_stream.stop()

    if new_px_cm is not None:
        print(f"\n[Success] ESP32 Scale factor updated: {new_px_cm:.2f} pixels per cm!")
        print(f"Run '.venv\\Scripts\\python.exe main_esp32.py' to test live scanning!")
    else:
        print("\n[Info] Calibration cancelled.")


if __name__ == "__main__":
    main()
