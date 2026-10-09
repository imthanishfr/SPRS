"""
Smart Packaging Recommendation System (SPRS) — ESP32 Camera Runner (main_esp32.py)

Supports:
1. Serial COM Port USB (e.g. COM4)
2. ESP32-CAM & Phone Wi-Fi Stream URLs (e.g. http://192.168.x.x:81/stream)
3. Camera Indices (0, 1, 2...)
"""

import sys
import os
import json
import time
import argparse
import cv2
import numpy as np
from typing import Union, Dict, Any, Optional, Tuple, List

import sprs
from sprs.async_pipeline import AsyncPackagingPipeline
from sprs.camera_stream import CameraStream

ESP32_USB_CONFIG_FILE = "esp32_usb_config.json"


def scan_external_usb_cameras(max_check=6) -> List[int]:
    """Scan for active camera video capture ports."""
    ports = []
    for idx in range(max_check):
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if not cap or not cap.isOpened():
            cap = cv2.VideoCapture(idx)
        if cap and cap.isOpened():
            ret, _ = cap.read()
            if ret:
                ports.append(idx)
            cap.release()
    return ports


def load_esp32_usb_source() -> Union[int, str]:
    """Load saved USB camera source / COM port / URL from esp32_usb_config.json."""
    if os.path.exists(ESP32_USB_CONFIG_FILE):
        try:
            with open(ESP32_USB_CONFIG_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                src = data.get("usb_source", "COM4")
                if isinstance(src, str) and src.isdigit():
                    return int(src)
                return src
        except Exception:
            pass
    return "COM4"


def save_esp32_usb_source(src: Union[int, str]):
    """Save camera source to esp32_usb_config.json."""
    try:
        with open(ESP32_USB_CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump({"usb_source": str(src)}, f, indent=2)
        print(f"[SPRS USB] Saved ESP32 camera source: '{src}' to '{ESP32_USB_CONFIG_FILE}'")
    except Exception as e:
        print(f"[SPRS USB Error] Could not save USB config: {e}")


def render_esp32_dashboard(
    camera_frame: np.ndarray,
    res: dict,
    fps: float,
    scan_mode: str,
    stored_flat_dims: dict,
    stored_height: float,
    canvas_w: int = 1280,
    canvas_h: int = 768
) -> np.ndarray:
    """
    Render ESP32 camera dashboard with 2-Step Tilt Scan status indicators.
    """
    canvas = np.full((canvas_h, canvas_w, 3), 18, dtype=np.uint8)

    # 1. TOP HEADER BAR
    cv2.rectangle(canvas, (0, 0), (canvas_w, 50), (24, 30, 38), -1)
    cv2.putText(canvas, "SPRS — ESP32 CAMERA PACKING MODULE & TILT SCANNER", (20, 33),
                cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 255, 200), 2)
                
    calib_str = f"Scale: {res.get('pixels_per_cm', 15.0):.1f} px/cm | FPS: {fps:.1f}"
    cv2.putText(canvas, calib_str, (canvas_w - 360, 33),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 220, 240), 1)

    # 2. LEFT PANEL: CAMERA STREAM (700 x 620 at (15, 65))
    cam_panel_w, cam_panel_h = 700, 620
    cam_x, cam_y = 15, 65

    cv2.rectangle(canvas, (cam_x, cam_y), (cam_x + cam_panel_w, cam_y + cam_panel_h), (35, 42, 52), -1)
    cv2.rectangle(canvas, (cam_x, cam_y), (cam_x + cam_panel_w, cam_y + cam_panel_h), (0, 200, 255), 2)

    if camera_frame is not None and camera_frame.size > 0:
        scaled_cam = cv2.resize(camera_frame, (cam_panel_w - 10, cam_panel_h - 10))
        canvas[cam_y + 5:cam_y + 5 + (cam_panel_h - 10), cam_x + 5:cam_x + 5 + (cam_panel_w - 10)] = scaled_cam

        step_str = f"SCAN STEP: [{scan_mode}] — Press SPACEBAR to switch"
        step_color = (0, 255, 255) if scan_mode == "STEP 1: FLAT SCAN (L x W)" else (255, 165, 0)
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

    # 3. RIGHT PANEL: PACKAGING DASHBOARD (535 x 620 at (730, 65))
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
        draw_section_header("ESP32 CAMERA ACTIVE")
        cy += 15
        cv2.putText(canvas, "Place object under ESP32 camera packing surface.", (panel_x + 20, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
    else:
        draw_section_header("1. 2-STEP SCAN METRICS (ESP32)")
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
        cy += 48

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
    cv2.putText(canvas, "[SPACEBAR] Toggle Tilt Step | [S] Snapshot | [C] Calibrate ESP32 | [Q] Quit",
                (25, 740), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)

    return canvas


def render_esp32_standby_dashboard(
    camera_src: Union[int, str],
    status_msg: str,
    canvas_w: int = 1280,
    canvas_h: int = 768
) -> np.ndarray:
    """Render a standby UI canvas when camera stream is connecting or frame is None."""
    canvas = np.full((canvas_h, canvas_w, 3), 20, dtype=np.uint8)

    # Header
    cv2.rectangle(canvas, (0, 0), (canvas_w, 50), (24, 30, 38), -1)
    cv2.putText(canvas, "SPRS — ESP32 CAMERA MODULE (CONNECTING...)", (20, 33),
                cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 255, 200), 2)

    # Center Panel Box
    cx, cy, cw, ch = 200, 130, 880, 520
    cv2.rectangle(canvas, (cx, cy), (cx + cw, cy + ch), (30, 38, 48), -1)
    cv2.rectangle(canvas, (cx, cy), (cx + cw, cy + ch), (0, 200, 255), 2)

    cv2.putText(canvas, "SEARCHING FOR ESP32 CAMERA VIDEO STREAM", (cx + 120, cy + 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)

    cv2.putText(canvas, f"Configured Source: '{camera_src}'", (cx + 40, cy + 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (220, 220, 240), 1)

    status_str = status_msg if status_msg else "Waiting for JPEG frames / Wi-Fi URL..."
    cv2.putText(canvas, f"Status: {status_str[:85]}", (cx + 40, cy + 135),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 255, 150), 1)

    # Instructions box inside standby
    ix, iy, iw, ih = cx + 30, cy + 165, cw - 60, 310
    cv2.rectangle(canvas, (ix, iy), (ix + iw, iy + ih), (20, 25, 33), -1)
    cv2.rectangle(canvas, (ix, iy), (ix + iw, iy + ih), (60, 80, 100), 1)

    cv2.putText(canvas, "CONNECTIVITY & TROUBLESHOOTING GUIDE:", (ix + 20, iy + 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 200), 2)

    instructions = [
        "1. Physical USB (COM4): ESP32 plugged via USB appears as COM4 port.",
        "2. Wi-Fi Stream: Standard ESP32 CameraWebServer streams over Wi-Fi (e.g. http://192.168.x.x:81/stream).",
        "3. If ESP32 prints Wi-Fi URL over Serial, SPRS auto-detects and connects to it!",
        "4. Press [U] to manually enter ESP32 Wi-Fi Stream URL or COM Port.",
        "5. Press [D] to enable DEMO MODE (simulates earbuds scan for full pipeline test).",
        "6. Press [Q] or [ESC] to quit."
    ]

    for idx, inst in enumerate(instructions):
        cv2.putText(canvas, inst, (ix + 20, iy + 75 + idx * 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 215, 230), 1)

    # Footer
    cv2.rectangle(canvas, (0, 695), (canvas_w, canvas_h), (15, 20, 26), -1)
    cv2.putText(canvas, "[U] Enter Stream URL / COM Port | [D] Toggle Demo Mode | [Q] Quit",
                (25, 740), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)

    return canvas


def create_demo_frame(width: int = 700, height: int = 620, tick: int = 0) -> np.ndarray:
    """Generate synthetic camera frame for testing when physical camera output is pending."""
    frame = np.full((height, width, 3), 40, dtype=np.uint8)
    cv2.rectangle(frame, (20, 20), (width - 20, height - 20), (70, 75, 80), -1)
    
    for x in range(20, width - 20, 50):
        cv2.line(frame, (x, 20), (x, height - 20), (85, 90, 95), 1)
    for y in range(20, height - 20, 50):
        cv2.line(frame, (20, y), (width - 20, y), (85, 90, 95), 1)

    ox1, oy1 = 200, 180
    ox2, oy2 = 500, 420
    cv2.rectangle(frame, (ox1, oy1), (ox2, oy2), (210, 180, 140), -1)
    cv2.rectangle(frame, (ox1, oy1), (ox2, oy2), (160, 120, 80), 3)

    cv2.putText(frame, "EARBUDS CASE (DEMO PRESET)", (ox1 + 35, oy1 + 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (40, 40, 40), 2)
    return frame


def main():
    parser = argparse.ArgumentParser(description="SPRS ESP32 Camera Runner")
    parser.add_argument("--device", type=str, default=None, help="Camera index (e.g. 1), COM Port (e.g. COM4), or IP URL")
    parser.add_argument("--url", type=str, default=None, help="ESP32 IP Stream URL (e.g. http://192.168.1.100:81/stream)")
    parser.add_argument("--demo", action="store_true", help="Start directly in Demo Mode")
    args = parser.parse_args()

    print("\n=======================================================")
    print("  SPRS — ESP32 CAMERA MODULE RUNNER")
    print("=======================================================\n")

    if args.url:
        camera_src = args.url
    elif args.device:
        camera_src = int(args.device) if args.device.isdigit() else args.device
    else:
        saved_src = load_esp32_usb_source()
        print(f"Saved Camera Source: {saved_src}")
        camera_src = saved_src

    save_esp32_usb_source(camera_src)

    window_name = "SPRS — ESP32 Camera Smart Packaging Dashboard"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 768)

    print("[SPRS AI Engine] Initializing PyTorch YOLOv8 + CLIP Zero-Shot Pipeline...")
    async_pipeline = AsyncPackagingPipeline()

    cam_stream = CameraStream(camera_src)
    cam_stream.start()

    scan_step_flat = True
    stored_flat_dims = {}
    stored_height = 0.0
    use_demo = args.demo

    step_count = 0
    prev_time = time.time()
    fps = 60.0

    try:
        while True:
            step_count += 1
            if use_demo:
                frame = create_demo_frame(700, 620, step_count)
                ret = True
            else:
                ret, frame = cam_stream.read()

            if not ret or frame is None:
                # Render Standby Dashboard non-blockingly to keep Windows OS GUI event loop pumping!
                standby_canvas = render_esp32_standby_dashboard(
                    camera_src=camera_src,
                    status_msg=cam_stream.status_message,
                    canvas_w=1280,
                    canvas_h=768
                )
                cv2.imshow(window_name, standby_canvas)
                key = cv2.waitKey(15) & 0xFF

                if key in [ord('q'), ord('Q'), 27]:
                    print("[SPRS Info] Exiting ESP32 camera runner.")
                    break
                elif key in [ord('d'), ord('D')]:
                    use_demo = not use_demo
                    print(f"[SPRS Info] Demo Mode set to: {use_demo}")
                elif key in [ord('u'), ord('U')]:
                    print("\n[SPRS Camera Setup] Enter ESP32 Stream URL (e.g. http://192.168.1.100:81/stream) or COM Port (e.g. COM4):")
                    new_src = input("New Camera Source: ").strip()
                    if new_src:
                        cam_stream.stop()
                        camera_src = int(new_src) if new_src.isdigit() else new_src
                        cam_stream = CameraStream(camera_src)
                        cam_stream.start()
                        save_esp32_usb_source(camera_src)
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

            dashboard = render_esp32_dashboard(
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
                print("[SPRS Info] Exiting ESP32 camera runner.")
                break
            elif key == 32:  # SPACEBAR
                scan_step_flat = not scan_step_flat
                curr_mode = "FLAT FOOTPRINT (L x W)" if scan_step_flat else "SIDE TILT PROFILE (H)"
                print(f"[SPRS 2-Step Scanner] Switched to mode: {curr_mode}")
            elif key in [ord('d'), ord('D')]:
                use_demo = not use_demo
                print(f"[SPRS Info] Toggled Demo Mode: {use_demo}")
            elif key in [ord('u'), ord('U')]:
                print("\n[SPRS Camera Setup] Enter ESP32 Stream URL or COM Port:")
                new_src = input("New Camera Source: ").strip()
                if new_src:
                    cam_stream.stop()
                    camera_src = int(new_src) if new_src.isdigit() else new_src
                    cam_stream = CameraStream(camera_src)
                    cam_stream.start()
                    save_esp32_usb_source(camera_src)
            elif key in [ord('s'), ord('S')]:
                print("\n-------------------------------------------------------")
                print("  SPRS ESP32 RECOMMENDATION SNAPSHOT LOG")
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
                    print(f"Camera Source: {camera_src}")
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
                print("-------------------------------------------------------\n")
            elif key in [ord('c'), ord('C')]:
                print("[SPRS Info] Opening Interactive Calibration tool for ESP32 Camera...")
                calib_res = async_pipeline.sync_pipeline.calibration_mgr.interactive_calibrate_live(cam_stream)
                if calib_res is not None:
                    async_pipeline.reload_calibration()
                    print(f"[SPRS Info] Updated camera scale to {calib_res:.2f} px/cm")

    finally:
        async_pipeline.stop()
        cam_stream.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

