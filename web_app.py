"""
SPRS Insane Web Application Backend (web_app.py)

FastAPI + MJPEG Video Stream + REST API
Serves live camera feed, 2-Step interactive scanning, and full analysis modal data.
"""

import os
import sys
import json
import time
import re
import base64
import cv2
import numpy as np
from typing import Dict, Any, Optional
from contextlib import asynccontextmanager
from fastapi import FastAPI, Response, Request
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from pydantic import BaseModel
import sprs
from sprs.async_pipeline import AsyncPackagingPipeline
from sprs.camera_stream import CameraStream

# Global System State
async_pipeline: Optional[AsyncPackagingPipeline] = None
cam_stream: Optional[CameraStream] = None

current_source = "COM4"
scan_state = {
    "step": 1,  # 1: Flat Scan, 2: Side Tilt Scan, 3: Analysis Modal Open
    "step_name": "STEP 1: FLAT SCAN (L x W)",
    "stored_flat_dims": None,
    "stored_height": 0.0,
    "final_analysis": None,
    "snapshot_base64": None
}


def sanitize_url(src: str) -> str:
    """Sanitize URL typos like double :8080:8080."""
    if isinstance(src, str):
        src = re.sub(r'(:8080)+', ':8080', src.strip())
        src = re.sub(r'(:81)+', ':81', src)
    return src


def initialize_sprs():
    global async_pipeline, cam_stream, current_source
    if async_pipeline is None:
        print("[SPRS Web] Initializing PyTorch AI Pipeline (YOLOv8 + CLIP)...")
        async_pipeline = AsyncPackagingPipeline()
        if async_pipeline.sync_pipeline.calibration_mgr.perspective_quad:
            quad = async_pipeline.sync_pipeline.calibration_mgr.perspective_quad
            async_pipeline.sync_pipeline.dimensioner.set_perspective_quad(quad)
    if cam_stream is None:
        raw_src = CameraStream.load_saved_source()
        current_source = sanitize_url(str(raw_src))
        print(f"[SPRS Web] Starting Camera Stream on source: '{current_source}'...")
        cam_stream = CameraStream(current_source)
        cam_stream.start()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    initialize_sprs()
    yield
    # Shutdown
    global async_pipeline, cam_stream
    if async_pipeline:
        async_pipeline.stop()
    if cam_stream:
        cam_stream.stop()


app = FastAPI(title="SPRS — Smart Packaging Recommendation System", lifespan=lifespan)

# Ensure static folder exists
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(status_code=204)


def generate_mjpeg_stream():
    """Generator function yielding MJPEG video frames with live bounding boxes."""
    global cam_stream, async_pipeline, scan_state

    while True:
        if cam_stream is None:
            time.sleep(0.05)
            continue

        ret, frame = cam_stream.read()
        
        if not ret or frame is None:
            standby = np.full((540, 720, 3), 20, dtype=np.uint8)
            cv2.putText(standby, "CONNECTING TO CAMERA STREAM...", (180, 260),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(standby, f"Source: {current_source}", (200, 300),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            _, encoded_img = cv2.imencode('.jpg', standby)
            try:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + encoded_img.tobytes() + b'\r\n')
            except Exception:
                break
            time.sleep(0.05)
            continue

        async_pipeline.update_frame(frame)
        rec = async_pipeline.get_latest_recommendation()

        # Render Bounding Box overlay on live frame
        annotated = frame.copy()

        # Render perspective quad overlay if active
        if rec.get("perspective_corrected", False) and rec.get("perspective_quad"):
            quad_pts = np.array(rec["perspective_quad"], np.int32).reshape((-1, 1, 2))
            cv2.polylines(annotated, [quad_pts], True, (255, 165, 0), 2)
            cv2.putText(annotated, "PERSPECTIVE RECTIFIED ROI", (rec["perspective_quad"][0][0] + 5, rec["perspective_quad"][0][1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 165, 0), 1)

        if rec.get("detected", False):
            x1, y1, x2, y2 = rec["bbox"]
            cat = rec["category_classification"]["category"]
            color = (0, 255, 0)
            if "Fragile" in cat:
                color = (0, 0, 255)
            elif "Electronics" in cat:
                color = (255, 165, 0)

            # Draw bounding box
            # Draw bounding box with thicker outline
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
            
            raw_dims = rec["raw_object_dimensions"]
            label = f"{cat} | {raw_dims['dimensions_str']}"
            if rec.get("perspective_corrected", False):
                label += " [PERSPECTIVE CORRECTED]"
            
            font_scale = 0.85
            thickness = 2
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            
            # High contrast black badge with colored accent border
            badge_y = max(th + 20, y1 - 12)
            cv2.rectangle(annotated, (x1, badge_y - th - 12), (x1 + tw + 20, badge_y + 8), (0, 0, 0), -1)
            cv2.rectangle(annotated, (x1, badge_y - th - 12), (x1 + tw + 20, badge_y + 8), color, 2)
            cv2.putText(annotated, label, (x1 + 10, badge_y - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

        _, encoded_img = cv2.imencode('.jpg', annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        try:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + encoded_img.tobytes() + b'\r\n')
        except Exception:
            break
        
        time.sleep(0.03)


@app.get("/video_feed")
def video_feed():
    """Video streaming endpoint returning MJPEG multipart feed."""
    return StreamingResponse(generate_mjpeg_stream(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/api/state")
def get_state():
    """Get current AI recommendation, camera state, and scan progress."""
    rec = async_pipeline.get_latest_recommendation() if async_pipeline else {}
    return {
        "camera_source": current_source,
        "scan_state": scan_state,
        "recommendation": rec
    }


@app.post("/api/calibrate_perspective")
async def calibrate_perspective(request: Request):
    """API endpoint to set or reset 4-point perspective homography quad."""
    global async_pipeline
    data = await request.json()
    quad = data.get("quad")
    reset = data.get("reset", False)

    if reset:
        if async_pipeline and async_pipeline.sync_pipeline:
            async_pipeline.sync_pipeline.dimensioner.set_perspective_quad(None)
            async_pipeline.sync_pipeline.calibration_mgr.save_config(
                async_pipeline.sync_pipeline.calibration_mgr.pixels_per_cm,
                perspective_quad=None
            )
        return {"status": "success", "message": "Perspective calibration reset."}

    if isinstance(quad, list) and len(quad) == 4:
        if async_pipeline and async_pipeline.sync_pipeline:
            async_pipeline.sync_pipeline.dimensioner.set_perspective_quad(quad)
            async_pipeline.sync_pipeline.calibration_mgr.save_config(
                async_pipeline.sync_pipeline.calibration_mgr.pixels_per_cm,
                perspective_quad=quad
            )
        return {"status": "success", "perspective_quad": quad}

    return {"status": "error", "message": "Invalid quad format. Expected 4 points [[x,y],...]."}


@app.post("/api/change_camera")
async def change_camera(request: Request):
    global cam_stream, current_source
    data = await request.json()
    new_src = sanitize_url(data.get("source", "").strip())
    if new_src:
        if cam_stream:
            cam_stream.stop()
        current_source = int(new_src) if new_src.isdigit() else new_src
        cam_stream = CameraStream(current_source)
        cam_stream.start()
        return {"status": "success", "source": current_source}
    return {"status": "error", "message": "Invalid camera source"}


@app.post("/api/capture_step1")
def capture_step1():
    """Capture Step 1 (Flat Footprint L x W)."""
    global scan_state, async_pipeline
    rec = async_pipeline.get_latest_recommendation()
    if not rec.get("detected", False):
        return {"status": "error", "message": "No object detected in camera view."}

    raw = rec["raw_object_dimensions"]
    scan_state["stored_flat_dims"] = {
        "length_cm": raw["length_cm"],
        "width_cm": raw["width_cm"]
    }
    scan_state["step"] = 2
    scan_state["step_name"] = "STEP 2: SIDE TILT SCAN (H)"
    
    return {
        "status": "success",
        "message": f"Footprint captured: {raw['length_cm']} x {raw['width_cm']} cm. Now tilt object for height and press ENTER.",
        "flat_dims": scan_state["stored_flat_dims"]
    }


class CalibrationRequest(BaseModel):
    camera_height_cm: float = 50.0
    ref_length_cm: float = 6.02
    ref_width_cm: float = 4.5
    ref_height_cm: float = 2.5
    custom_pixels_per_cm: Optional[float] = None


class AutoCalibrateRequest(BaseModel):
    camera_height_cm: float = 20.0
    ref_length_cm: float = 6.02
    ref_width_cm: float = 4.5
    ref_height_cm: float = 2.5


@app.post("/api/auto_calibrate_from_camera")
def auto_calibrate_from_camera(req: AutoCalibrateRequest):
    """
    Detects the calibration test piece in the live camera feed and
    computes the exact pixel-to-cm ratio so the measured object matches
    the user's specified reference dimensions (e.g. 6.02 cm).
    """
    global async_pipeline, cam_stream
    if not async_pipeline or not async_pipeline.sync_pipeline:
        return {"status": "error", "message": "AI Pipeline initializing..."}

    ret, frame = cam_stream.read() if cam_stream else (False, None)
    if not ret or frame is None:
        return {"status": "error", "message": "Could not capture live camera frame."}

    det = async_pipeline.sync_pipeline.dimensioner.detect_and_measure(frame)
    if not det.get("detected", False):
        return {
            "status": "error",
            "message": "No test piece detected in camera view. Please place your test piece (e.g. earbuds box) on the packing surface under the camera."
        }

    bbox = det["bbox_int"]
    x1, y1, x2, y2 = bbox
    pixel_w = abs(x2 - x1)
    pixel_h = abs(y2 - y1)
    max_pixel_dim = max(pixel_w, pixel_h)

    target_ref_l = max(0.1, req.ref_length_cm)
    computed_px_cm = round(max_pixel_dim / target_ref_l, 2)

    async_pipeline.sync_pipeline.calibration_mgr.save_config(
        pixels_per_cm=computed_px_cm,
        camera_height_cm=req.camera_height_cm
    )
    async_pipeline.sync_pipeline.dimensioner.pixels_per_cm = computed_px_cm

    return {
        "status": "success",
        "detected_pixel_dim": max_pixel_dim,
        "ref_length_cm": target_ref_l,
        "pixels_per_cm": computed_px_cm,
        "measured_dimensions_str": f"{round(pixel_w / computed_px_cm, 1)} × {round(pixel_h / computed_px_cm, 1)} cm"
    }


@app.post("/api/calibrate_custom")
def calibrate_custom(req: CalibrationRequest):
    """Update camera calibration from the calibration page."""
    global async_pipeline
    if not async_pipeline or not async_pipeline.sync_pipeline:
        return {"status": "error", "message": "Pipeline initializing..."}

    px_cm = req.custom_pixels_per_cm if (req.custom_pixels_per_cm and req.custom_pixels_per_cm > 0) else round(1570.0 / max(10.0, req.camera_height_cm), 2)
    async_pipeline.sync_pipeline.calibration_mgr.save_config(
        pixels_per_cm=px_cm,
        camera_height_cm=req.camera_height_cm
    )
    async_pipeline.sync_pipeline.dimensioner.pixels_per_cm = px_cm
    return {
        "status": "success",
        "pixels_per_cm": px_cm,
        "camera_height_cm": req.camera_height_cm,
        "ref_dims": {
            "length_cm": req.ref_length_cm,
            "width_cm": req.ref_width_cm,
            "height_cm": req.ref_height_cm
        }
    }


@app.post("/api/capture_step2")
def capture_step2():
    """Capture Step 2 (Side Tilt Height H) and generate final analysis modal data."""
    global scan_state, async_pipeline, cam_stream
    rec = async_pipeline.get_latest_recommendation()
    if not rec.get("detected", False):
        return {"status": "error", "message": "No object detected in camera view."}

    raw = rec["raw_object_dimensions"]
    if scan_state["stored_flat_dims"] is None:
        scan_state["stored_flat_dims"] = {
            "length_cm": raw["length_cm"],
            "width_cm": raw["width_cm"]
        }

    flat_l = scan_state["stored_flat_dims"]["length_cm"]
    flat_w = scan_state["stored_flat_dims"]["width_cm"]
    tilt_h = raw["width_cm"]  # Height from side tilt profile width

    # Re-run bin packer & advanced metrics calculation with final 2-Step (L, W, H)
    category = rec["category_classification"]["category"]
    packing_res = async_pipeline.sync_pipeline.bin_packer.pack(
        length_cm=flat_l,
        width_cm=flat_w,
        height_cm=tilt_h,
        padding_buffer_cm=rec["packaging_rules"]["padding_buffer_cm"]
    )

    from sprs.rules import compute_comprehensive_packaging_metrics
    adv_metrics = compute_comprehensive_packaging_metrics(
        category=category,
        obj_l=flat_l,
        obj_w=flat_w,
        obj_h=tilt_h,
        optimal_box=packing_res["optimal_custom_box"],
        stock_box=packing_res["best_stock_box"]
    )

    # Capture high-res crop base64 image for modal panel
    snapshot_b64 = None
    ret, frame = cam_stream.read()
    if ret and frame is not None:
        crop = rec.get("crop", frame)
        _, buffer = cv2.imencode('.jpg', crop, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        snapshot_b64 = base64.b64encode(buffer).decode('utf-8')

    final_analysis = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "yolo_class": rec["yolo_class"],
        "category": category,
        "category_confidence": rec["category_classification"]["confidence"],
        "dimensions": {
            "length_cm": flat_l,
            "width_cm": flat_w,
            "height_cm": tilt_h,
            "volume_cm3": round(flat_l * flat_w * tilt_h, 1),
            "dimensions_str": f"{flat_l} × {flat_w} × {tilt_h} cm"
        },
        "packaging_rules": rec["packaging_rules"],
        "optimal_custom_box": packing_res["optimal_custom_box"],
        "best_stock_box": packing_res["best_stock_box"],
        "space_utilization_pct": packing_res["space_utilization_pct"],
        "advanced_metrics": adv_metrics,
        "snapshot_base64": snapshot_b64
    }

    scan_state["step"] = 3
    scan_state["step_name"] = "ANALYSIS & PACKAGING PANEL OPEN"
    scan_state["final_analysis"] = final_analysis
    scan_state["snapshot_base64"] = snapshot_b64

    return {
        "status": "success",
        "analysis": final_analysis
    }


@app.post("/api/reset_scan")
def reset_scan():
    """Reset scan workflow to Step 1."""
    global scan_state
    scan_state = {
        "step": 1,
        "step_name": "STEP 1: FLAT SCAN (L x W)",
        "stored_flat_dims": None,
        "stored_height": 0.0,
        "final_analysis": None,
        "snapshot_base64": None
    }
    return {"status": "success"}


@app.get("/calibrate", response_class=HTMLResponse)
def calibrate_page():
    """Serve dedicated camera calibration page."""
    html_file = os.path.join(STATIC_DIR, "calibrate.html")
    if os.path.exists(html_file):
        with open(html_file, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read(), headers={"Cache-Control": "no-cache, no-store, must-revalidate"})
    return "<h1>SPRS Calibration Page Loading...</h1>"


@app.get("/", response_class=HTMLResponse)
def index_page():
    """Serve main web application index.html."""
    html_file = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(html_file):
        with open(html_file, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read(), headers={"Cache-Control": "no-cache, no-store, must-revalidate"})
    return "<h1>SPRS Web App Loading...</h1>"


if __name__ == "__main__":
    import uvicorn
    import sys

    port = 8000
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        port = int(sys.argv[1])

    print("\n=======================================================")
    print("  SPRS INDUSTRIAL WEB DASHBOARD")
    print(f"  Open browser: http://localhost:{port}")
    print("=======================================================\n")
    uvicorn.run(app, host="0.0.0.0", port=port)

