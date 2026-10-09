"""
Master Integration Pipeline (sprs/pipeline.py)

Glues together:
1. Webcam/Frame Acquisition -> YOLOv8 Object Detection & Bbox Extraction
2. Dimensioning -> Real-world L x W x H via pixel-to-cm calibration
3. Zero-Shot Classifier -> CLIP Category matching (6 categories)
4. Rules Table -> Cardboard type, padding buffer, filling material, instructions
5. Bin-Packing Engine -> Optimal custom box & Best warehouse stock box selection
"""

import time
import numpy as np
from typing import Dict, Any, Optional

from sprs.dimensioner import ObjectDimensioner
from sprs.classifier import ClipCategoryClassifier
from sprs.rules import get_packaging_rules
from sprs.bin_packer import PackingEngine
from sprs.calibration import CalibrationManager


class SmartPackagingPipeline:
    def __init__(
        self,
        yolo_model: str = "yolov8n.pt",
        clip_model: str = "openai/clip-vit-base-patch32",
        stock_inventory: Optional[list] = None
    ):
        self.calibration_mgr = CalibrationManager()
        self.dimensioner = ObjectDimensioner(
            model_weights=yolo_model,
            pixels_per_cm=self.calibration_mgr.pixels_per_cm
        )
        self.classifier = ClipCategoryClassifier(model_name=clip_model)
        self.bin_packer = PackingEngine(stock_inventory=stock_inventory)

    def reload_calibration(self):
        """Reload calibration factor from config."""
        new_px_cm = self.calibration_mgr.load_config()
        self.dimensioner.update_calibration(new_px_cm)

    def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Process a single image frame through the end-to-end recommendation pipeline.
        
        Args:
            frame: OpenCV BGR image array
            
        Returns:
            Comprehensive recommendation result dictionary.
        """
        start_time = time.time()

        det_result = self.dimensioner.detect_and_measure(frame)

        if not det_result.get("detected", False):
            return {
                "detected": False,
                "reason": det_result.get("reason", "No object detected"),
                "process_time_ms": round((time.time() - start_time) * 1000, 1)
            }

        crop = det_result["crop"]
        yolo_cls = det_result["yolo_class"]
        dims = det_result["dimensions"]

        category, cat_conf, cat_scores = self.classifier.classify(
            image_input=crop,
            heuristic_label=yolo_cls
        )

        from sprs.rules import get_packaging_rules, compute_comprehensive_packaging_metrics
        rules = get_packaging_rules(category)
        buffer_cm = float(rules.get("padding_buffer_cm", 2.0))

        packing_result = self.bin_packer.pack(
            length_cm=dims["length_cm"],
            width_cm=dims["width_cm"],
            height_cm=dims["height_cm"],
            padding_buffer_cm=buffer_cm
        )

        metrics = compute_comprehensive_packaging_metrics(
            category=category,
            obj_l=dims["length_cm"],
            obj_w=dims["width_cm"],
            obj_h=dims["height_cm"],
            optimal_box=packing_result["optimal_custom_box"],
            stock_box=packing_result["best_stock_box"]
        )

        elapsed_ms = round((time.time() - start_time) * 1000, 1)

        return {
            "detected": True,
            "yolo_class": yolo_cls,
            "detection_confidence": det_result["confidence"],
            "bbox": det_result["bbox_int"],
            "raw_object_dimensions": {
                "length_cm": dims["length_cm"],
                "width_cm": dims["width_cm"],
                "height_cm": dims["height_cm"],
                "volume_cm3": dims["volume_cm3"],
                "dimensions_str": f"{dims['length_cm']} × {dims['width_cm']} × {dims['height_cm']} cm"
            },
            "category_classification": {
                "category": category,
                "confidence": cat_conf,
                "all_category_scores": cat_scores
            },
            "packaging_rules": {
                "cardboard_type": rules["cardboard_type"],
                "padding_buffer_cm": buffer_cm,
                "filling_material": rules["filling_material"],
                "handling_instructions": rules["handling_instructions"],
                "orientation_recommendation": rules["orientation_recommendation"]
            },
            "optimal_custom_box": packing_result["optimal_custom_box"],
            "best_stock_box": packing_result["best_stock_box"],
            "optimal_utilization_pct": packing_result.get("optimal_utilization_pct", packing_result["optimal_custom_box"].get("optimal_utilization_pct", 0.0)),
            "stock_utilization_pct": packing_result.get("stock_utilization_pct", packing_result["space_utilization_pct"]),
            "space_utilization_pct": packing_result["space_utilization_pct"],
            "fit_status": packing_result["status"],
            "advanced_metrics": metrics,
            "pixels_per_cm": det_result["pixels_per_cm"],
            "process_time_ms": elapsed_ms
        }

