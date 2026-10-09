"""
SPRS Mobile Webcam Launcher (run_demo.py)
"""

import sys
import os
import argparse
import subprocess

def main():
    parser = argparse.ArgumentParser(description="SPRS Mobile Launcher")
    parser.add_argument("--calibrate", action="store_true", help="Launch Mobile Camera Calibration Tool")
    parser.add_argument("--demo", action="store_true", help="Launch SPRS in synthetic demo mode")
    parser.add_argument("--test", action="store_true", help="Run automated test suite")

    args = parser.parse_args()

    python_exe = os.path.join(".venv", "Scripts", "python.exe")
    if not os.path.exists(python_exe):
        python_exe = sys.executable

    if args.calibrate:
        print("[SPRS Launcher] Launching Mobile Camera Calibration Tool...")
        subprocess.run([python_exe, "calibrate.py"])
    elif args.test:
        print("[SPRS Launcher] Running unit test suite...")
        subprocess.run([python_exe, "test_sprs.py"])
    elif args.demo:
        print("[SPRS Launcher] Launching synthetic video demo mode...")
        subprocess.run([python_exe, "main.py", "--demo"])
    else:
        # Interactive Menu
        print("\n=======================================================")
        print("  SMART PACKAGING RECOMMENDATION SYSTEM (SPRS)")
        print("  Mobile Phone IP Camera / Webcam Module")
        print("=======================================================\n")
        print("Choose an option:")
        print("  [1] Launch Mobile Camera / Webcam Dashboard (main.py)")
        print("  [2] Calibrate Mobile Camera Scale (calibrate.py)")
        print("  [3] Run Synthetic Video Demo")
        print("  [4] Run Automated Unit Tests (test_sprs.py)")
        
        try:
            choice = input("\nEnter choice (1-4) [Default: 1 - Mobile Cam]: ").strip()
            if choice == "2":
                subprocess.run([python_exe, "calibrate.py"])
            elif choice == "3":
                subprocess.run([python_exe, "main.py", "--demo"])
            elif choice == "4":
                subprocess.run([python_exe, "test_sprs.py"])
            else:
                subprocess.run([python_exe, "main.py"])
        except KeyboardInterrupt:
            print("\nExiting launcher.")

if __name__ == "__main__":
    main()

