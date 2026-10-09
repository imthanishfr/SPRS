"""
Camera Pixel-to-Centimeter Calibration Manager (sprs/calibration.py)

Exclusive Reference Object: Earbuds Case (Length: 6.02 cm / 60.2 mm)
Camera Mounting Height: 20.0 cm above packing surface

Saves, loads, and manages calibration profiles for camera heights and pixel-to-cm ratios.
Supports interactive live video stream 2-point calibration in OpenCV.
"""

import json
import os
import cv2
import numpy as np
from typing import Dict, Any, Optional, List

CONFIG_FILE = "calibration_config.json"
EARBUDS_CASE_LENGTH_CM = 6.02  # 60.2 mm


class CalibrationManager:
    def __init__(self, config_path: str = CONFIG_FILE):
        self.config_path = config_path
        self.pixels_per_cm: float = 15.0
        self.camera_height_cm: float = 20.0
        self.perspective_quad: Optional[List[List[float]]] = None
        self.load_config()

    def load_config(self) -> float:
        """Load calibration config from JSON file if exists."""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.pixels_per_cm = float(data.get("pixels_per_cm", 15.0))
                    self.camera_height_cm = float(data.get("camera_height_cm", 20.0))
                    quad = data.get("perspective_quad")
                    if isinstance(quad, list) and len(quad) == 4:
                        self.perspective_quad = [[float(p[0]), float(p[1])] for p in quad]
                    print(f"[SPRS Calibration] Loaded scale: {self.pixels_per_cm:.2f} px/cm from {self.config_path}")
                    if self.perspective_quad:
                        print(f"[SPRS Calibration] Perspective Homography Quad active.")
            except Exception as e:
                print(f"[SPRS Calibration Warning] Could not load config: {e}")
        return self.pixels_per_cm

    def set_web_calibration(
        self,
        camera_height_cm: float,
        ref_length_cm: float = 10.0,
        custom_pixels_per_cm: Optional[float] = None
    ) -> float:
        """
        Set calibration based on camera height and reference object inputs from web wizard.
        """
        self.camera_height_cm = max(10.0, float(camera_height_cm))
        
        if custom_pixels_per_cm and custom_pixels_per_cm > 0:
            new_px_cm = float(custom_pixels_per_cm)
        else:
            # Baseline optical height compensation formula (focal length constant ~ 1570 px)
            # px_per_cm = focal_length_px / camera_height_cm
            focal_length_px = 1570.0
            new_px_cm = round(focal_length_px / self.camera_height_cm, 2)

        self.save_config(new_px_cm, self.camera_height_cm)
        return self.pixels_per_cm

    def save_config(
        self,
        pixels_per_cm: float,
        camera_height_cm: Optional[float] = None,
        perspective_quad: Optional[List[List[float]]] = None
    ) -> bool:
        """Save calibration data to JSON file."""
        self.pixels_per_cm = round(pixels_per_cm, 2)
        if camera_height_cm is not None:
            self.camera_height_cm = round(camera_height_cm, 1)
        if perspective_quad is not None:
            self.perspective_quad = perspective_quad

        try:
            data = {
                "pixels_per_cm": self.pixels_per_cm,
                "camera_height_cm": self.camera_height_cm,
                "perspective_quad": self.perspective_quad
            }
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            print(f"[SPRS Calibration] Saved calibration data: {self.pixels_per_cm} px/cm, height: {self.camera_height_cm} cm to '{self.config_path}'.")
            return True
        except Exception as e:
            print(f"[SPRS Calibration Error] Could not save calibration config: {e}")
            return False

    def interactive_calibrate_live(self, cam_stream) -> Optional[float]:
        """
        Interactive live video stream 2-point calibration tool in OpenCV.
        Continuously displays live camera feed while user clicks 2 points along earbuds case (6.02 cm).
        """
        points = []

        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                if len(points) < 2:
                    points.append((x, y))

        window_name = "SPRS Live Calibration — Earbuds Case (6.02 cm)"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1024, 600)
        cv2.setMouseCallback(window_name, mouse_callback)

        print("\n=== SPRS Live Camera Scale Calibration Mode ===")
        print(f"Reference Object: Earbuds Case (Length: {EARBUDS_CASE_LENGTH_CM} cm)")
        print("Click START & END points along the 6.02 cm edge of your earbuds case.")
        print("Press 'ENTER' or 'C' to save scale, 'R' to reset points, or 'Q' / ESC to cancel.\n")

        known_length_cm = EARBUDS_CASE_LENGTH_CM

        while True:
            ret, frame = cam_stream.read()
            if not ret or frame is None:
                cv2.waitKey(10)
                continue

            canvas = frame.copy()

            # Instruction Overlay Header
            cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 70), (20, 25, 32), -1)
            cv2.putText(canvas, f"CALIBRATION MODE: Earbuds Case Length ({known_length_cm} cm)", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(canvas, "Click Start & End points along earbuds case length | Press ENTER to confirm", (20, 58),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)

            # Draw clicked points
            for i, p in enumerate(points):
                cv2.circle(canvas, p, 6, (0, 0, 255), -1)
                cv2.putText(canvas, f"P{i+1}", (p[0] + 8, p[1] - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

            if len(points) == 2:
                p1, p2 = points[0], points[1]
                cv2.line(canvas, p1, p2, (0, 255, 0), 2)
                pixel_dist = float(np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2))
                computed_px_cm = pixel_dist / known_length_cm
                
                # Feedback Tag
                cv2.rectangle(canvas, (20, 80), (450, 120), (15, 20, 25), -1)
                cv2.putText(canvas, f"Measured: {pixel_dist:.1f} px => {computed_px_cm:.2f} px/cm", (30, 108),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

            cv2.imshow(window_name, canvas)
            key = cv2.waitKey(20) & 0xFF

            if key in [13, ord('c'), ord('C')]:  # ENTER or C
                if len(points) == 2:
                    p1, p2 = points[0], points[1]
                    pixel_dist = float(np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2))
                    computed_px_cm = pixel_dist / known_length_cm
                    self.save_config(computed_px_cm)
                    cv2.destroyWindow(window_name)
                    return computed_px_cm
            elif key in [ord('r'), ord('R')]:
                points.clear()
            elif key in [27, ord('q'), ord('Q')]:  # ESC or Q
                cv2.destroyWindow(window_name)
                return None
