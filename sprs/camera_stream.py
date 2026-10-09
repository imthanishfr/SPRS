"""
Zero-Latency Camera & Stream Reader (sprs/camera_stream.py)

Supports:
1. Serial COM Ports (COM4) for ESP32 USB-to-Serial boards
2. ESP32-CAM & Phone Wi-Fi Streams (http://192.168.x.x:81/stream or http://192.168.x.x:8080/video)
3. DirectShow USB Webcams / Virtual Cameras (Index 0, 1, 2)
"""

import os
import json
import time
import re
import threading
import cv2
import numpy as np
from typing import Union, Tuple, Optional

CONFIG_FILE = "camera_config.json"


class CameraStream:
    def __init__(self, camera_source: Union[int, str] = 0, baudrate: int = 115200):
        self.camera_source = self._parse_source(camera_source)
        self.baudrate = baudrate
        self.cap: Optional[cv2.VideoCapture] = None
        self.serial_port = None
        self.lock = threading.Lock()
        self.running = False
        self.latest_frame: Optional[np.ndarray] = None
        self.status_message: str = ""
        self.worker_thread: Optional[threading.Thread] = None

    @staticmethod
    def _parse_source(src: Union[int, str]) -> Union[int, str]:
        if isinstance(src, str):
            src_clean = src.strip()
            if src_clean.isdigit():
                return int(src_clean)
            return src_clean
        return src

    @classmethod
    def load_saved_source(cls) -> Union[int, str]:
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    src = data.get("camera_source", 0)
                    return cls._parse_source(src)
            except Exception:
                pass
        return 0

    @classmethod
    def save_source(cls, camera_source: Union[int, str]):
        try:
            with open(CONFIG_FILE, "w", encoding="utf-8") as f:
                json.dump({"camera_source": str(camera_source)}, f, indent=2)
            print(f"[SPRS Camera] Saved camera config: '{camera_source}' to '{CONFIG_FILE}'")
        except Exception as e:
            print(f"[SPRS Camera Error] Could not save camera config: {e}")

    def start(self) -> bool:
        src_str = str(self.camera_source).upper()
        print(f"[SPRS Camera] Opening stream source: '{self.camera_source}'...")
        self.status_message = f"Connecting to source: '{self.camera_source}'..."

        if src_str.startswith("COM"):
            try:
                import serial
                print(f"[SPRS Camera] Connecting to Serial USB Port '{self.camera_source}' at {self.baudrate} baud...")
                self.serial_port = serial.Serial(self.camera_source, self.baudrate, timeout=1)
                self.running = True
                self.status_message = f"Serial Port '{self.camera_source}' connected. Waiting for frames or Wi-Fi URL..."
                self.worker_thread = threading.Thread(target=self._grab_serial_frames, daemon=True)
                self.worker_thread.start()
                self.save_source(self.camera_source)
                time.sleep(0.3)
                print(f"[SPRS Camera] Serial COM port '{self.camera_source}' connected.")
                return True
            except Exception as e:
                err_msg = f"Could not open Serial Port '{self.camera_source}': {e}"
                print(f"[SPRS Camera Error] {err_msg}")
                self.status_message = err_msg
                return False

        if isinstance(self.camera_source, str) and (self.camera_source.startswith("http") or self.camera_source.startswith("rtsp")):
            url = self.camera_source
            if not url.endswith("/stream") and not url.endswith("/video") and ":" not in url.split("//")[1]:
                # Try default ESP32-CAM stream port 81
                url = f"{url}:81/stream"
                print(f"[SPRS Camera] Formatted URL to ESP32 stream: '{url}'")
            self.cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        elif isinstance(self.camera_source, int):
            self.cap = cv2.VideoCapture(self.camera_source, cv2.CAP_DSHOW)
            if not self.cap or not self.cap.isOpened():
                self.cap = cv2.VideoCapture(self.camera_source)
        else:
            self.cap = cv2.VideoCapture(self.camera_source)

        if not self.cap or not self.cap.isOpened():
            err_msg = f"Failed to open camera source '{self.camera_source}'."
            print(f"[SPRS Camera Error] {err_msg}")
            self.status_message = err_msg
            return False

        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.running = True
        self.status_message = f"Stream Connected: '{self.camera_source}'"
        self.worker_thread = threading.Thread(target=self._grab_cv_frames, daemon=True)
        self.worker_thread.start()

        self.save_source(self.camera_source)
        time.sleep(0.3)
        print("[SPRS Camera] Camera stream successfully connected.")
        return True

    def _grab_cv_frames(self):
        while self.running and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret and frame is not None:
                with self.lock:
                    self.latest_frame = frame
            else:
                time.sleep(0.01)

    def _grab_serial_frames(self):
        """Background thread reading JPEG frames or Serial log lines from COM port."""
        buffer = bytearray()
        auto_cap: Optional[cv2.VideoCapture] = None

        while self.running:
            # 1. If auto_cap (Wi-Fi stream) was opened via Serial URL discovery
            if auto_cap is not None and auto_cap.isOpened():
                ret, frame = auto_cap.read()
                if ret and frame is not None:
                    with self.lock:
                        self.latest_frame = frame
                    time.sleep(0.01)
                    continue

            # 2. Read raw bytes from serial port
            if self.serial_port and self.serial_port.is_open:
                try:
                    chunk = self.serial_port.read(1024)
                    if chunk:
                        buffer.extend(chunk)

                        # Check for printed text lines (e.g. WiFi IP output)
                        try:
                            text_line = chunk.decode("utf-8", errors="ignore")
                            if "http://" in text_line or "https://" in text_line:
                                match = re.search(r'https?://[0-9\.:]+[^\s]*', text_line)
                                if match:
                                    found_url = match.group(0).rstrip(".'\"")
                                    print(f"\n[SPRS Camera] ESP32 Wi-Fi Stream URL detected on Serial COM port: '{found_url}'")
                                    self.status_message = f"Detected ESP32 Wi-Fi Stream: {found_url}"
                                    if auto_cap is None:
                                        print(f"[SPRS Camera] Auto-connecting to detected Wi-Fi stream: '{found_url}'...")
                                        auto_cap = cv2.VideoCapture(found_url, cv2.CAP_FFMPEG)
                        except Exception:
                            pass

                        # Search for JPEG start (0xFF 0xD8) and end (0xFF 0xD9)
                        start = buffer.find(b'\xff\xd8')
                        end = buffer.find(b'\xff\xd9')

                        if start != -1 and end != -1 and end > start:
                            jpg_bytes = buffer[start:end+2]
                            buffer = buffer[end+2:]

                            np_arr = np.frombuffer(jpg_bytes, dtype=np.uint8)
                            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                            if frame is not None:
                                with self.lock:
                                    self.latest_frame = frame
                        elif len(buffer) > 65536:
                            buffer = buffer[-4096:]
                except Exception:
                    time.sleep(0.01)

            time.sleep(0.01)

        if auto_cap and auto_cap.isOpened():
            auto_cap.release()

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        with self.lock:
            if self.latest_frame is not None:
                return True, self.latest_frame.copy()
        return False, None

    def stop(self):
        self.running = False
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=1.0)
        if self.cap and self.cap.isOpened():
            self.cap.release()
        if self.serial_port and self.serial_port.is_open:
            self.serial_port.close()
        print("[SPRS Camera] Camera stream closed.")

