# ⛑️ Real-Time Safety Helmet & Gear Detector

> Computer vision pipeline to detect helmet compliance on campus and industrial zones using YOLO object detection.

[![Status: Blueprint Starter](https://img.shields.io/badge/Status-Starter_Blueprint-brightgreen.svg?style=flat-square)](#)
[![Domain: Computer Vision](https://img.shields.io/badge/Domain-Computer_Vision-purple.svg?style=flat-square)](#)

---

## 🎯 Problem Statement
Two-wheeler rider safety and laboratory/workshop helmet compliance require automated monitoring to ensure safety standard adherence.

## 💡 Solution
- **Model Architecture:** Ultralytics YOLO (YOLOv8 / YOLOv11) trained on custom annotated helmet/head datasets.
- **Inference:** OpenCV video capture stream with bounding boxes and compliance counter.

## 🚀 Quickstart

### Option 1: Run Instantly (Zero Dependencies)
```bash
python detect.py
```

### Option 2: Live Webcam YOLO Inference
```bash
pip install -r requirements.txt
python detect.py
```

## 🤝 Open Contributions (Good First Issues)
- [ ] Add an annotated dataset loader script (Roboflow / Kaggle integration).
- [ ] Implement audio alert buzzer sound when non-compliance is detected.
- [ ] Export trained YOLO weights to ONNX format for browser-based WebAssembly inference.
