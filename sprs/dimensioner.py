"""
Object Detection & Real-World Dimensioning (sprs/dimensioner.py)

Uses YOLOv8 (yolov8n.pt) for object detection and bounding box extraction.
Converts bounding box pixel dimensions into real-world centimeters using fixed camera height calibration.
Includes temporal exponential moving average (EMA) smoothing for stability.
"""

import cv2
import numpy as np
from typing import Dict, Tuple, Any, Optional, List

DEFAULT_PIXELS_PER_CM = 15.0
DEFAULT_CAMERA_HEIGHT_CM = 60.0


class ObjectDimensioner:
    def __init__(
        self,
        model_weights: str = "yolov8n.pt",
        pixels_per_cm: float = DEFAULT_PIXELS_PER_CM,
        conf_threshold: float = 0.20,
        smoothing_factor: float = 0.3,
        perspective_quad: Optional[List[List[float]]] = None
    ):
        self.model_weights = model_weights
        self.pixels_per_cm = pixels_per_cm
        self.conf_threshold = conf_threshold
        self.smoothing_factor = smoothing_factor
        self.model = None
        self.initialized = False
        
        self.perspective_quad: Optional[List[List[float]]] = None
        self.M: Optional[np.ndarray] = None
        self.M_inv: Optional[np.ndarray] = None
        self.dst_size: Optional[Tuple[int, int]] = None

        self.smoothed_bbox: Optional[Tuple[float, float, float, float]] = None
        self._load_yolo()
        
        if perspective_quad:
            self.set_perspective_quad(perspective_quad)

    def set_perspective_quad(self, quad_pts: Optional[List[List[float]]]):
        """
        Set 4-point quadrilateral for Homography Perspective Rectification.
        Order: Top-Left, Top-Right, Bottom-Right, Bottom-Left.
        """
        if quad_pts is None or len(quad_pts) != 4:
            self.perspective_quad = None
            self.M = None
            self.M_inv = None
            self.dst_size = None
            return

        pts = np.array(quad_pts, dtype="float32")
        (tl, tr, br, bl) = pts

        width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        max_w = max(int(width_a), int(width_b))

        height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        max_h = max(int(height_a), int(height_b))

        dst = np.array([
            [0, 0],
            [max_w - 1, 0],
            [max_w - 1, max_h - 1],
            [0, max_h - 1]
        ], dtype="float32")

        self.M = cv2.getPerspectiveTransform(pts, dst)
        self.M_inv = cv2.getPerspectiveTransform(dst, pts)
        self.dst_size = (max_w, max_h)
        self.perspective_quad = quad_pts
        print(f"[SPRS Dimensioner] Perspective Homography Matrix calculated ({max_w}x{max_h} px).")

    def _load_yolo(self):
        """Lazy load YOLOv8 model."""
        try:
            from ultralytics import YOLO
            print(f"[SPRS Dimensioner] Loading YOLOv8 model '{self.model_weights}'...")
            self.model = YOLO(self.model_weights)
            self.initialized = True
            print("[SPRS Dimensioner] YOLOv8 successfully loaded.")
        except Exception as e:
            print(f"[SPRS Dimensioner Warning] Could not load YOLOv8: {e}")
            print("[SPRS Dimensioner Warning] Falling back to OpenCV contour detector.")
            self.initialized = False

    def update_calibration(self, pixels_per_cm: float):
        """Update pixels-to-cm calibration factor."""
        if pixels_per_cm > 0:
            self.pixels_per_cm = pixels_per_cm
            print(f"[SPRS Dimensioner] Updated calibration factor: {pixels_per_cm:.2f} px/cm")

    def detect_and_measure(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Detect object in frame and compute real-world dimensions with optional perspective warping.
        """
        if frame is None or frame.size == 0:
            return {"detected": False, "error": "Invalid frame"}

        if self.M is not None and self.dst_size is not None:
            # Warp frame to rectified top-down bird's-eye view
            rectified_frame = cv2.warpPerspective(frame, self.M, self.dst_size)
            h_rect, w_rect = rectified_frame.shape[:2]

            res = (
                self._detect_yolo(rectified_frame, h_rect, w_rect)
                if (self.initialized and self.model is not None)
                else self._detect_contour_fallback(rectified_frame, h_rect, w_rect)
            )

            if res.get("detected", False):
                res["perspective_corrected"] = True
                res["perspective_quad"] = self.perspective_quad
            return res

        h_frame, w_frame = frame.shape[:2]
        if self.initialized and self.model is not None:
            return self._detect_yolo(frame, h_frame, w_frame)
        else:
            return self._detect_contour_fallback(frame, h_frame, w_frame)

    def _detect_yolo(self, frame: np.ndarray, h_frame: int, w_frame: int) -> Dict[str, Any]:
        """YOLOv8 object detection flow with background table surface filtering."""
        try:
            results = self.model(frame, conf=self.conf_threshold, verbose=False)
            
            best_box = None
            max_conf = 0.0
            best_cls_name = "object"

            if len(results) > 0 and len(results[0].boxes) > 0:
                boxes = results[0].boxes
                names = results[0].names

                for box in boxes:
                    xyxy = box.xyxy[0].cpu().numpy()
                    cls_id = int(box.cls[0].cpu().numpy())
                    conf = float(box.conf[0].cpu().numpy())
                    
                    x1, y1, x2, y2 = xyxy
                    bw = x2 - x1
                    bh = y2 - y1
                    area = bw * bh
                    
                    # Filter out boxes covering > 45% of frame (table background) or touching outer border
                    if area > (w_frame * h_frame * 0.45) or area < 400:
                        continue
                    if x1 <= 10 or y1 <= 10 or x2 >= (w_frame - 10) or y2 >= (h_frame - 10):
                        continue

                    if conf > max_conf:
                        max_conf = conf
                        best_box = (float(x1), float(y1), float(x2), float(y2))
                        best_cls_name = names.get(cls_id, "object")

            if best_box is not None:
                return self._process_bbox(frame, best_box, best_cls_name, max_conf, h_frame, w_frame)
        
        except Exception as e:
            print(f"[SPRS Dimensioner Error] YOLO detection error: {e}")

        return self._detect_contour_fallback(frame, h_frame, w_frame)

    def _detect_contour_fallback(self, frame: np.ndarray, h_frame: int, w_frame: int) -> Dict[str, Any]:
        """
        Fallback contour & edge detector isolating physical items on table surface.
        Filters out background table surface gradients & outer frame borders.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 1. Edge map via Canny
        edges = cv2.Canny(blurred, 30, 150)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        # 2. Otsu threshold maps
        _, thresh1 = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, thresh2 = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        best_box = None
        best_score = -1.0

        center_x, center_y = w_frame / 2.0, h_frame / 2.0

        for map_img in [closed_edges, thresh1, thresh2]:
            contours, _ = cv2.findContours(map_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                
                # Rule 1: Exclude noise (< 600 px) or background table contours (> 45% of frame)
                if area < 600 or area > (w_frame * h_frame * 0.45):
                    continue

                x, y, w, h = cv2.boundingRect(cnt)
                
                # Rule 2: Exclude contours touching outer frame border (margin of 15px)
                if x <= 15 or y <= 15 or (x + w) >= (w_frame - 15) or (y + h) >= (h_frame - 15):
                    continue

                # Rule 3: Score based on area compactness & closeness to center of view
                box_center_x = x + w / 2.0
                box_center_y = y + h / 2.0
                dist_from_center = np.hypot(box_center_x - center_x, box_center_y - center_y)
                max_diag = np.hypot(w_frame, h_frame)
                
                norm_dist = 1.0 - (dist_from_center / (max_diag / 2.0))
                score = (area / (w_frame * h_frame)) * 0.6 + max(0.0, norm_dist) * 0.4

                if score > best_score:
                    best_score = score
                    best_box = (float(x), float(y), float(x + w), float(y + h))

        if best_box is not None:
            return self._process_bbox(frame, best_box, "object", 0.75, h_frame, w_frame)

        return {"detected": False, "reason": "No clear item detected on table surface"}

    def _process_bbox(
        self,
        frame: np.ndarray,
        raw_bbox: Tuple[float, float, float, float],
        cls_name: str,
        conf: float,
        h_frame: int,
        w_frame: int
    ) -> Dict[str, Any]:
        """Apply EMA temporal smoothing and compute real-world cm dimensions."""
        x1, y1, x2, y2 = raw_bbox

        if self.smoothed_bbox is None:
            self.smoothed_bbox = raw_bbox
        else:
            sx1, sy1, sx2, sy2 = self.smoothed_bbox
            alpha = self.smoothing_factor
            self.smoothed_bbox = (
                sx1 * (1 - alpha) + x1 * alpha,
                sy1 * (1 - alpha) + y1 * alpha,
                sx2 * (1 - alpha) + x2 * alpha,
                sy2 * (1 - alpha) + y2 * alpha
            )

        sx1, sy1, sx2, sy2 = self.smoothed_bbox
        ix1, iy1, ix2, iy2 = max(0, int(sx1)), max(0, int(sy1)), min(w_frame, int(sx2)), min(h_frame, int(sy2))

        crop = frame[iy1:iy2, ix1:ix2]
        if crop.size == 0:
            crop = frame

        w_px = max(1.0, float(ix2 - ix1))
        h_px = max(1.0, float(iy2 - iy1))

        dim1_cm = round(w_px / self.pixels_per_cm, 1)
        dim2_cm = round(h_px / self.pixels_per_cm, 1)

        length_cm = max(dim1_cm, dim2_cm)
        width_cm = min(dim1_cm, dim2_cm)
        height_cm = round(max(1.5, width_cm * 0.45), 1)

        return {
            "detected": True,
            "bbox_int": (ix1, iy1, ix2, iy2),
            "bbox_float": (sx1, sy1, sx2, sy2),
            "crop": crop,
            "yolo_class": cls_name,
            "confidence": round(conf, 2),
            "pixels_per_cm": self.pixels_per_cm,
            "dimensions": {
                "length_cm": length_cm,
                "width_cm": width_cm,
                "height_cm": height_cm,
                "volume_cm3": round(length_cm * width_cm * height_cm, 2)
            }
        }
