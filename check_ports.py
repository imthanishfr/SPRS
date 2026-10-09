"""
SPRS Hardware Diagnostic Utility (check_ports.py)

Scans:
1. Windows OpenCV Camera Devices (Index 0, 1, 2...)
2. Active Windows Serial COM Ports (COM1, COM2, COM3...) for USB FTDI/CH340 ESP32 boards
"""

import cv2
import sys

def main():
    print("\n=======================================================")
    print("  SPRS HARDWARE DIAGNOSTIC UTILITY")
    print("=======================================================\n")

    # 1. Scan OpenCV Video Capture Devices
    print("[1] Scanning Windows OpenCV Camera Video Devices...")
    found_cams = []
    for idx in range(5):
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if not cap or not cap.isOpened():
            cap = cv2.VideoCapture(idx)
        if cap and cap.isOpened():
            ret, _ = cap.read()
            if ret:
                found_cams.append(idx)
            cap.release()

    if found_cams:
        print(f"    -> Active Camera Video Devices Found: Indices {found_cams}")
        for c_idx in found_cams:
            if c_idx == 0:
                print(f"       • Index 0: Built-in Laptop Webcam")
            else:
                print(f"       • Index {c_idx}: External USB Video Device")
    else:
        print("    -> No Camera Video Devices found.")

    # 2. Scan Serial COM Ports (USB-to-UART FTDI/CH340/CP2102)
    print("\n[2] Scanning Windows Serial COM Ports (ESP32 USB-Serial Boards)...")
    com_ports = []
    try:
        import serial.tools.list_ports
        ports = serial.tools.list_ports.comports()
        for p in ports:
            com_ports.append(f"{p.device} - {p.description}")
    except ImportError:
        # Fallback manual COM port test
        import serial
        for i in range(1, 20):
            port_name = f"COM{i}"
            try:
                s = serial.Serial(port_name)
                com_ports.append(f"{port_name} (Active)")
                s.close()
            except Exception:
                pass

    if com_ports:
        print("    -> Found connected Serial COM Ports:")
        for cp in com_ports:
            print(f"       • {cp}")
    else:
        print("    -> No Serial COM Ports detected.")

    print("\n-------------------------------------------------------")
    print("  DIAGNOSTIC SUMMARY:")
    print("-------------------------------------------------------")
    if len(found_cams) == 1 and 0 in found_cams:
        print("  Notice: Windows ONLY detects 1 video device (Index 0 = Laptop Webcam).")
        print("  Your ESP32-CAM is connected via USB-Serial (FTDI/CH340/CP2102).")
        print("  ESP32-CAM requires Wi-Fi IP stream (http://192.168.x.x:81/stream)")
        print("  OR Serial JPEG bytes over COM port.")
    print("-------------------------------------------------------\n")

if __name__ == "__main__":
    main()
