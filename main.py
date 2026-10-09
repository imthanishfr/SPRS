"""
Smart Packaging Recommendation System (SPRS) — Main Entry Point (main.py)

Features:
- Multi-Threaded Async GPU Pipeline (NVIDIA CUDA)
- High-Performance Zero-Latency Camera Stream (Supports Phone IP Streams & Webcams)
- 1280x768 Side-by-Side UI Dashboard (Zero text overlap)
- Real-time YOLOv8 Object Bounding Box Tracking & Pixel-to-cm Dimensioning
- Zero-Shot CLIP Category Classification
- Rules-based Packaging & Filling Material recommendation
- 3D Bin-Packing matching: Optimal Custom Box vs Best Warehouse Stock Box Match
"""

import sys
import time
import argparse
import cv2
import numpy as np

import sprs
from sprs.async_pipeline import AsyncPackagingPipeline
from sprs.camera_stream import CameraStream


def create_demo_frame(frame_width: int = 720, frame_height: int = 540, step_count: int = 0) -> np.ndarray:
    """Generate synthetic camera frame for demonstration mode."""
    frame = np.full((frame_height, frame_width, 3), 32, dtype=np.uint8)
    
    cv2.rectangle(frame, (40, 40), (frame_width - 40, frame_height - 40), (48, 48, 48), -1)
    cv2.rectangle(frame, (40, 40), (frame_width - 40, frame_height - 40), (70, 70, 70), 2)
    
    for i in range(80, frame_width - 40, 80):
        cv2.line(frame, (i, 40), (i, frame_height - 40), (55, 55, 55), 1)
    for j in range(80, frame_height - 40, 80):
        cv2.line(frame, (40, j), (frame_width - 40, j), (55, 55, 55), 1)

    offset = int(np.sin(step_count * 0.08) * 8)
    x1, y1 = 220 + offset, 140
    x2, y2 = 500 + offset, 380
    
    cv2.rectangle(frame, (x1, y1), (x2, y2), (180, 140, 60), -1)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (230, 190, 90), 3)
    cv2.putText(frame, "TEST ELECTRONIC GADGET", (x1 + 15, y1 + 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
                
    return frame


def render_dashboard(
    camera_frame: np.ndarray,
    res: dict,
    fps: float,
    scan_mode: str = "STEP 1: FLAT SCAN (L x W)",
    stored_flat_dims: dict = None,
    stored_height: float = 0.0,
    canvas_w: int = 1280,
    canvas_h: int = 768
) -> np.ndarray:
    """
    Render modern side-by-side dashboard layout with 2-Step Tilt Scan metrics.
    """
    canvas = np.full((canvas_h, canvas_w, 3), 18, dtype=np.uint8)

    # 1. TOP HEADER BAR
    cv2.rectangle(canvas, (0, 0), (canvas_w, 50), (24, 30, 38), -1)
    cv2.putText(canvas, "SMART PACKAGING RECOMMENDATION SYSTEM (SPRS)", (20, 33),
                cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 255, 200), 2)
                
    calib_str = f"Scale: {res.get('pixels_per_cm', 15.0):.1f} px/cm | FPS: {fps:.1f}"
    cv2.putText(canvas, calib_str, (canvas_w - 360, 33),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 220, 240), 1)

    # 2. LEFT PANEL: CAMERA STREAM CONTAINER
    cam_panel_w, cam_panel_h = 700, 620
    cam_x, cam_y = 15, 65

    cv2.rectangle(canvas, (cam_x, cam_y), (cam_x + cam_panel_w, cam_y + cam_panel_h), (35, 42, 52), -1)
    cv2.rectangle(canvas, (cam_x, cam_y), (cam_x + cam_panel_w, cam_y + cam_panel_h), (0, 200, 255), 2)

    if camera_frame is not None and camera_frame.size > 0:
        scaled_cam = cv2.resize(camera_frame, (cam_panel_w - 10, cam_panel_h - 10))
        canvas[cam_y + 5:cam_y + 5 + (cam_panel_h - 10), cam_x + 5:cam_x + 5 + (cam_panel_w - 10)] = scaled_cam

        step_str = f"SCAN STEP: [{scan_mode}] — Press SPACEBAR to switch"
        step_color = (0, 255, 255) if "STEP 1" in scan_mode else (255, 165, 0)
        cv2.rectangle(canvas, (cam_x + 10, cam_y + 10), (cam_x + cam_panel_w - 10, cam_y + 45), (15, 20, 28), -1)
        cv2.rectangle(canvas, (cam_x + 10, cam_y + 10), (cam_x + cam_panel_w - 10, cam_y + 45), step_color, 1)
        cv2.putText(canvas, step_str, (cam_x + 20, cam_y + 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, step_color, 2)

        if res.get("detected", False):
            orig_h, orig_w = camera_frame.shape[:2]
            scale_x = (cam_panel_w - 10) / orig_w
            scale_y = (cam_panel_h - 10) / orig_h

            ox1, oy1, ox2, oy2 = res["bbox"]
            bx1 = int(cam_x + 5 + ox1 * scale_x)
            by1 = int(cam_y + 5 + oy1 * scale_y)
            bx2 = int(cam_x + 5 + ox2 * scale_x)
            by2 = int(cam_y + 5 + oy2 * scale_y)

            category = res["category_classification"]["category"]
            box_color = (0, 255, 0)
            if "Fragile" in category:
                box_color = (0, 0, 255)
            elif "Electronics" in category:
                box_color = (255, 165, 0)

            line_len = 20
            thickness = 3
            cv2.line(canvas, (bx1, by1), (bx1 + line_len, by1), box_color, thickness)
            cv2.line(canvas, (bx1, by1), (bx1, by1 + line_len), box_color, thickness)
            cv2.line(canvas, (bx2, by1), (bx2 - line_len, by1), box_color, thickness)
            cv2.line(canvas, (bx2, by1), (bx2, by1 + line_len), box_color, thickness)
            cv2.line(canvas, (bx1, by2), (bx1 + line_len, by2), box_color, thickness)
            cv2.line(canvas, (bx1, by2), (bx1, by2 - line_len), box_color, thickness)
            cv2.line(canvas, (bx2, by2), (bx2 - line_len, by2), box_color, thickness)
            cv2.line(canvas, (bx2, by2), (bx2, by2 - line_len), box_color, thickness)

            cv2.rectangle(canvas, (bx1, by1), (bx2, by2), box_color, 1)

            raw_dims = res["raw_object_dimensions"]
            tag_str = f"{category} | {raw_dims['dimensions_str']}"
            (tw, th), _ = cv2.getTextSize(tag_str, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(canvas, (bx1, max(cam_y + 55, by1 - 22)), (bx1 + tw + 10, by1), box_color, -1)
            cv2.putText(canvas, tag_str, (bx1 + 5, max(cam_y + 70, by1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # 3. RIGHT PANEL: PACKAGING DASHBOARD
    panel_x, panel_y = 730, 65
    panel_w, panel_h = 535, 620

    cv2.rectangle(canvas, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (25, 32, 42), -1)
    cv2.rectangle(canvas, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (0, 200, 255), 2)

    cy = panel_y + 30

    def draw_section_header(text: str):
        nonlocal cy
        cv2.putText(canvas, text, (panel_x + 15, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 200), 2)
        cy += 24

    def draw_kv(title: str, val: str, val_color=(255, 255, 255), title_w=170):
        nonlocal cy
        cv2.putText(canvas, f"{title}:", (panel_x + 20, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (180, 195, 210), 1)
        cv2.putText(canvas, str(val), (panel_x + 20 + title_w, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, val_color, 2)
        cy += 24

    if not res.get("detected", False):
        draw_section_header("SYSTEM STATUS: WAITING FOR ITEM")
        cy += 15
        cv2.putText(canvas, "Place object under packing camera surface.", (panel_x + 20, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
    else:
        draw_section_header("1. OBJECT METRICS & CLASSIFICATION")
        draw_kv("YOLO Object Class", f"{res['yolo_class']} ({res['detection_confidence']*100:.0f}%)")
        
        raw_dims = res["raw_object_dimensions"]
        if stored_flat_dims and stored_height > 0:
            l_val = stored_flat_dims.get("length_cm", raw_dims["length_cm"])
            w_val = stored_flat_dims.get("width_cm", raw_dims["width_cm"])
            h_val = stored_height
            dim_str = f"{l_val} × {w_val} × {h_val} cm (2-Step Validated)"
            vol_val = round(l_val * w_val * h_val, 1)
        else:
            dim_str = raw_dims["dimensions_str"]
            vol_val = raw_dims["volume_cm3"]

        draw_kv("Real Dimensions", dim_str, val_color=(0, 255, 255))
        draw_kv("Object Volume", f"{vol_val:,.1f} cm3")
        
        cat_info = res["category_classification"]
        draw_kv("CLIP Category", f"{cat_info['category']} ({cat_info['confidence']*100:.0f}%)", val_color=(0, 255, 255))

        cy += 8
        cv2.line(canvas, (panel_x + 15, cy), (panel_x + panel_w - 15, cy), (60, 75, 95), 1)
        cy += 18

        draw_section_header("2. PROTECTIVE REQUIREMENTS")
        rules = res["packaging_rules"]
        draw_kv("Cardboard Grade", rules["cardboard_type"])
        draw_kv("Padding Buffer", f"+{rules['padding_buffer_cm']} cm per side")
        draw_kv("Filling Material", rules["filling_material"], val_color=(100, 255, 120))
        draw_kv("Orientation", rules["orientation_recommendation"])

        cy += 4
        inst_str = rules["handling_instructions"]
        cv2.rectangle(canvas, (panel_x + 15, cy), (panel_x + panel_w - 15, cy + 32), (45, 20, 25), -1)
        cv2.rectangle(canvas, (panel_x + 15, cy), (panel_x + panel_w - 15, cy + 32), (0, 0, 255), 1)
        cv2.putText(canvas, f"INSTRUCTION: {inst_str[:42]}", (panel_x + 22, cy + 21),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (120, 210, 255), 1)
        cy += 42

        cv2.line(canvas, (panel_x + 15, cy), (panel_x + panel_w - 15, cy), (60, 75, 95), 1)
        cy += 18

        draw_section_header("3. OPTIMAL BOX & STOCK MATCH")
        opt = res["optimal_custom_box"]
        draw_kv("Optimal Custom Box", opt["dimensions_str"], val_color=(255, 215, 0))
        draw_kv("Custom Box Vol", f"{opt['volume_cm3']:,.1f} cm3")

        stk = res["best_stock_box"]
        if stk:
            draw_kv("Warehouse Stock Box", f"{stk['box_id']} ({stk['name']})")
            draw_kv("Stock Box Dims", stk["dimensions_str"])
            
            util_pct = res["space_utilization_pct"]
            draw_kv("Space Efficiency", f"{util_pct}% Volume Utilization", val_color=(0, 255, 0))
            
            cy += 4
            bar_x = panel_x + 20
            bar_w = panel_w - 40
            bar_h = 16
            cv2.rectangle(canvas, (bar_x, cy), (bar_x + bar_w, cy + bar_h), (40, 50, 65), -1)
            fill_w = int(bar_w * (min(100.0, util_pct) / 100.0))
            fill_color = (0, 255, 0) if util_pct > 50 else (0, 200, 255)
            cv2.rectangle(canvas, (bar_x, cy), (bar_x + fill_w, cy + bar_h), fill_color, -1)
            cv2.rectangle(canvas, (bar_x, cy), (bar_x + bar_w, cy + bar_h), (120, 140, 160), 1)
            cy += 25
        else:
            draw_kv("Warehouse Stock Box", "OVERSIZED — MANUAL PACKING REQUIRED", val_color=(0, 0, 255))

    # 4. BOTTOM FOOTER BAR
    cv2.rectangle(canvas, (0, 695), (canvas_w, canvas_h), (15, 20, 26), -1)
    cv2.putText(canvas, "[SPACEBAR] Toggle Tilt Step | [S] Snapshot | [C] Calibrate Camera | [D] Demo Mode | [Q] Quit",
                (25, 740), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220, 220, 220), 1)

    return canvas


def main():
    parser = argparse.ArgumentParser(description="Smart Packaging Recommendation System (SPRS)")
    parser.add_argument("--camera", type=str, default=None, help="Camera index (e.g. 0, 1) or IP Stream URL (e.g. http://192.168.1.50:8080/video)")
    parser.add_argument("--demo", action="store_true", help="Run in synthetic video demo mode")
    args = parser.parse_args()

    print("\n=======================================================")
    print("  SMART PACKAGING RECOMMENDATION SYSTEM (SPRS)")
    print("  Mobile Camera / Webcam Runner & 2-Step Tilt Scanner")
    print("=======================================================\n")

    # Determine Camera Source
    if args.camera is not None:
        cam_src = args.camera
    else:
        cam_src = CameraStream.load_saved_source()

    async_pipeline = AsyncPackagingPipeline()
    use_demo = args.demo
    cam_stream = None

    if not use_demo:
        cam_stream = CameraStream(cam_src)
        if not cam_stream.start():
            print(f"\n[SPRS Notice] Could not connect to camera source '{cam_src}'.")
            print("Enter your phone's IP Stream URL shown in the IP Webcam app (e.g. http://192.168.1.50:8080/video).")
            user_ip = input("Enter Phone IP Stream URL or Camera Index [Press ENTER for Demo Mode]: ").strip()
            if user_ip:
                if not user_ip.startswith("http") and not user_ip.isdigit():
                    user_ip = f"http://{user_ip}:8080/video"
                cam_src = int(user_ip) if user_ip.isdigit() else user_ip
                cam_stream = CameraStream(cam_src)
                if not cam_stream.start():
                    print(f"[SPRS Warning] Connection to '{cam_src}' failed. Switching to Demo Mode.")
                    use_demo = True
            else:
                use_demo = True

    window_name = "SPRS — Smart Packaging Dashboard"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 768)

    scan_step_flat = True
    stored_flat_dims = {}
    stored_height = 0.0

    step_count = 0
    prev_time = time.time()
    fps = 60.0

    try:
        while True:
            step_count += 1
            if use_demo or cam_stream is None:
                frame = create_demo_frame(720, 540, step_count)
            else:
                ret, frame = cam_stream.read()
                if not ret or frame is None:
                    standby = np.full((768, 1280, 3), 20, dtype=np.uint8)
                    cv2.putText(standby, "CONNECTING TO MOBILE CAMERA STREAM...", (320, 380),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
                    cv2.putText(standby, "Press [D] for Demo Mode or [Q] to Quit", (400, 420),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
                    cv2.imshow(window_name, standby)
                    key = cv2.waitKey(15) & 0xFF
                    if key in [ord('q'), ord('Q'), 27]:
                        break
                    elif key in [ord('d'), ord('D')]:
                        use_demo = True
                    continue

            async_pipeline.update_frame(frame)
            recommendation = async_pipeline.get_latest_recommendation()

            if recommendation.get("detected", False):
                raw = recommendation["raw_object_dimensions"]
                if scan_step_flat:
                    stored_flat_dims = {"length_cm": raw["length_cm"], "width_cm": raw["width_cm"]}
                    scan_mode_name = "STEP 1: FLAT SCAN (L x W)"
                else:
                    stored_height = raw["width_cm"]
                    scan_mode_name = "STEP 2: SIDE TILT SCAN (H)"
            else:
                scan_mode_name = "STEP 1: FLAT SCAN (L x W)" if scan_step_flat else "STEP 2: SIDE TILT SCAN (H)"

            curr_time = time.time()
            dt = curr_time - prev_time
            prev_time = curr_time
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt)

            dashboard = render_dashboard(
                frame,
                recommendation,
                fps,
                scan_mode_name,
                stored_flat_dims,
                stored_height,
                1280,
                768
            )

            cv2.imshow(window_name, dashboard)
            key = cv2.waitKey(15) & 0xFF

            if key in [ord('q'), ord('Q'), 27]:
                print("[SPRS Info] Exiting application.")
                break
            elif key == 32:  # SPACEBAR
                scan_step_flat = not scan_step_flat
                curr_mode = "FLAT FOOTPRINT (L x W)" if scan_step_flat else "SIDE TILT PROFILE (H)"
                print(f"[SPRS 2-Step Scanner] Switched to mode: {curr_mode}")
            elif key in [ord('s'), ord('S')]:
                print("\n-------------------------------------------------------")
                print("  SPRS RECOMMENDATION SNAPSHOT LOG")
                print("-------------------------------------------------------")
                if recommendation.get("detected", False):
                    raw = recommendation["raw_object_dimensions"]
                    opt = recommendation["optimal_custom_box"]
                    stk = recommendation["best_stock_box"]
                    rules = recommendation["packaging_rules"]
                    cat = recommendation["category_classification"]

                    l_f = stored_flat_dims.get("length_cm", raw["length_cm"])
                    w_f = stored_flat_dims.get("width_cm", raw["width_cm"])
                    h_f = stored_height if stored_height > 0 else raw["height_cm"]

                    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
                    print(f"2-Step Dimensions: {l_f} x {w_f} x {h_f} cm")
                    print(f"Category Classified: {cat['category']} (Confidence: {cat['confidence']*100:.1f}%)")
                    print(f"Cardboard Type: {rules['cardboard_type']}")
                    print(f"Padding Buffer: +{rules['padding_buffer_cm']} cm per side")
                    print(f"Filling Material: {rules['filling_material']}")
                    print(f"Handling Instruction: {rules['handling_instructions']}")
                    print(f"Optimal Custom Box: {opt['dimensions_str']} (Vol: {opt['volume_cm3']} cm3)")
                    if stk:
                        print(f"Warehouse Stock Box: {stk['box_id']} - {stk['name']} ({stk['dimensions_str']})")
                        print(f"Space Utilization: {recommendation['space_utilization_pct']}%")
                    else:
                        print("Warehouse Stock Box: OVERSIZED — Manual packing required")
                else:
                    print("No object currently detected in snapshot frame.")
                print("-------------------------------------------------------\n")
            elif key in [ord('c'), ord('C')]:
                print("[SPRS Info] Opening Interactive Calibration tool...")
                calib_res = async_pipeline.sync_pipeline.calibration_mgr.interactive_calibrate_live(cam_stream)
                if calib_res is not None:
                    async_pipeline.reload_calibration()
                    print(f"[SPRS Info] Updated pipeline calibration to {calib_res:.2f} px/cm")
            elif key in [ord('d'), ord('D')]:
                use_demo = not use_demo
                print(f"[SPRS Info] Toggled Demo Mode: {use_demo}")

    finally:
        async_pipeline.stop()
        if cam_stream:
            cam_stream.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

