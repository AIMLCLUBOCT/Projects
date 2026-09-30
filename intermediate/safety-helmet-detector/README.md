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
```bash
pip install -r requirements.txt
python detect.py
```

## 🤝 Open Contributions (Good First Issues)
- [ ] Add an annotated dataset loader script (Roboflow integration).
- [ ] Add real-time webcam inference loop with FPS counter.
- [ ] Add audio alert when non-compliance is detected.
