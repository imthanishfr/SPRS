"""
SPRS Live Camera Calibration Utility (calibrate.py)

Exclusive Reference Object: Earbuds Case (Length: 60.2 mm / 6.02 cm)
Camera Mounting Height: 20.0 cm above packing surface
"""

import sys
import time
import cv2
from sprs.calibration import CalibrationManager, EARBUDS_CASE_LENGTH_CM
from sprs.camera_stream import CameraStream

def main():
    print("\n=======================================================")
    print("  SPRS LIVE CAMERA SCALE CALIBRATION UTILITY")
    print(f"  Reference Object: Earbuds Case (Length: {EARBUDS_CASE_LENGTH_CM} cm / 60.2 mm)")
    print("  Camera Mounting Height: 20.0 cm above packing surface")
    print("=======================================================\n")
    
    calib_mgr = CalibrationManager()
    print(f"Current Scale: {calib_mgr.pixels_per_cm:.2f} px/cm")

    cam_src = CameraStream.load_saved_source()
    cam_input = input(f"\nEnter Camera IP Stream URL or Index [Default: {cam_src}]: ").strip()
    if cam_input:
        if not cam_input.startswith("http") and not cam_input.isdigit():
            cam_input = f"http://{cam_input}:8080/video"
        cam_src = cam_input

    cam_stream = CameraStream(cam_src)
    if not cam_stream.start():
        print(f"[Error] Cannot open camera stream '{cam_src}'.")
        return

    print(f"\nLive camera stream connected: '{cam_src}'.")
    print("1. Place your earbuds case flat under your phone camera (Height: 20 cm).")
    print("2. Click START and END points along the 6.02 cm edge of your earbuds case.")
    print("3. Press ENTER or 'C' to save calibration scale.\n")

    new_px_cm = calib_mgr.interactive_calibrate_live(cam_stream)
    cam_stream.stop()

    if new_px_cm is not None:
        print(f"\n[Success] Scale factor updated: {new_px_cm:.2f} pixels per cm!")
        print(f"Run '.venv\\Scripts\\python.exe main.py' to test live scanning!")
    else:
        print("\n[Info] Calibration cancelled.")

if __name__ == "__main__":
    main()
