"""
Asynchronous Threaded Integration Pipeline (sprs/async_pipeline.py)

Decouples webcam frame acquisition & OpenCV UI rendering (60+ FPS) from heavy
GPU AI inference (YOLOv8 + CLIP zero-shot classification) running on a background worker thread.
"""

import time
import threading
import numpy as np
from typing import Dict, Any, Optional

from sprs.pipeline import SmartPackagingPipeline


class AsyncPackagingPipeline:
    def __init__(
        self,
        yolo_model: str = "yolov8n.pt",
        clip_model: str = "openai/clip-vit-base-patch32",
        stock_inventory: Optional[list] = None
    ):
        self.sync_pipeline = SmartPackagingPipeline(
            yolo_model=yolo_model,
            clip_model=clip_model,
            stock_inventory=stock_inventory
        )
        
        self.lock = threading.Lock()
        self.running = True
        
        self.latest_frame: Optional[np.ndarray] = None
        self.latest_result: Dict[str, Any] = {"detected": False, "reason": "Initializing AI Pipeline..."}
        self.new_frame_event = threading.Event()
        
        # Start background worker thread
        self.worker_thread = threading.Thread(target=self._ai_worker_loop, daemon=True)
        self.worker_thread.start()

    def update_frame(self, frame: np.ndarray):
        """Pass latest camera frame to background worker non-blockingly."""
        if frame is None or frame.size == 0:
            return
        with self.lock:
            self.latest_frame = frame.copy()
        self.new_frame_event.set()

    def get_latest_recommendation(self) -> Dict[str, Any]:
        """Fetch current AI recommendation result instantaneously (0ms delay)."""
        with self.lock:
            return self.latest_result.copy()

    def reload_calibration(self):
        """Reload calibration factor."""
        self.sync_pipeline.reload_calibration()

    def _ai_worker_loop(self):
        """Background thread loop running YOLO + CLIP AI inference continuously."""
        print("[SPRS Async Pipeline] AI Worker Thread started on GPU/CPU.")
        
        while self.running:
            # Wait until a new frame is ready
            self.new_frame_event.wait(timeout=0.1)
            self.new_frame_event.clear()
            
            frame_to_process = None
            with self.lock:
                if self.latest_frame is not None:
                    frame_to_process = self.latest_frame.copy()

            if frame_to_process is not None:
                # Run synchronous AI pipeline
                res = self.sync_pipeline.process_frame(frame_to_process)
                with self.lock:
                    self.latest_result = res
            
            time.sleep(0.01)

    def stop(self):
        """Stop background worker thread."""
        self.running = False
        self.new_frame_event.set()
