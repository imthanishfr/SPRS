# SPRS — Smart Packaging & Recommendation System 📦⚡

An industrial-grade computer vision and PyTorch AI system designed for real-world warehouse logistics, e-commerce fulfillment, and automated package optimization. **SPRS** measures physical object dimensions in real time, classifies items with zero-shot AI, determines optimal custom box specs, matches warehouse stock box inventory, and calculates protective fill & carbon metrics.

---

## Key Features 🚀

- **Pre-Scan Camera Calibration Wizard (`/calibrate`)**: Dedicated optical calibration page for camera distance ($H_{\text{cam}}$) and reference test piece verification.
- **2-Step 3D Dimensioning Engine**: Measures object Length ($L$), Width ($W$), and Height ($H$) in centimeters with sub-centimeter accuracy.
- **Dual-AI Category Classification**: Fuses YOLOv8 object detection with OpenAI CLIP zero-shot vision classification for high-precision category identification.
- **3D Bin-Packing & Warehouse Inventory Matching**: Evaluates 6 spatial rotations against a 14-tier stock box catalog to maximize volume utilization.
- **Dual Space Utilization Efficiency Gauges**: Separate percentage ring meters for **Optimal Custom Box** vs **Warehouse Stock Box** with dynamic color feedback (Red / Yellow / Green).
- **Interactive 3D Three.js Packing Inspector**: 520px Three.js WebGL viewport allowing operators to inspect cardboard layers, items, and protective void fill.

---

## Project Structure 📂

```
SPRS/
├── sprs/
│   ├── __init__.py
│   ├── async_pipeline.py     # Multi-threaded PyTorch AI worker thread
│   ├── bin_packer.py         # 3D Bin-packing & stock box inventory engine
│   ├── calibration.py        # Camera pixel-to-cm calibration manager
│   ├── camera_stream.py     # Fast OpenCV MJPEG camera stream handler
│   ├── classifier.py        # YOLO + CLIP zero-shot category classifier
│   ├── dimensioner.py        # OpenCV computer vision dimensioning engine
│   ├── pipeline.py          # Synchronous packaging pipeline orchestrator
│   └── rules.py             # Industrial packaging rules & density metrics
├── static/
│   ├── app.js               # Frontend UI controller & Three.js 3D visualizer
│   ├── calibrate.html       # Dedicated Camera Calibration Wizard page
│   ├── index.html           # Main Industrial Dashboard page
│   └── style.css            # Dark mode glassmorphism design system
├── web_app.py               # FastAPI backend & streaming REST API
├── main.py                  # CLI demonstration entrypoint
└── README.md
```

---

## Quick Start 🛠️

### 1. Prerequisites
- Python 3.10+
- PyTorch & OpenCV
- Web Camera or IP Camera Stream

### 2. Installation
```bash
# Clone the repository
git clone https://github.com/imthanishfr/SPRS.git
cd SPRS

# Create & activate virtual environment
python -m venv .venv
.venv\Scripts\activate  # On Windows
source .venv/bin/activate  # On Linux/macOS

# Install dependencies
pip install fastapi uvicorn opencv-python torch torchvision transformers pillow pydantic numpy
```

### 3. Running the Dashboard
```bash
python web_app.py 8000
```
Open your browser at **http://localhost:8000**.

---

## License 📜
MIT License — Free for commercial and research use.
