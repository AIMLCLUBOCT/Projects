# 🟡 Intermediate Projects

> Practical deep learning, computer vision, and NLP implementations with modular code and API endpoints.

---

## 🎯 Focus Areas
- Computer vision: Image classification, filtering, and real-time object detection using OpenCV and YOLO.
- Natural language processing: Text classification, Named Entity Recognition (NER), and sentiment analysis with Hugging Face Transformers.
- Asynchronous API serving with FastAPI and Pydantic schema validation.

---

## 📂 Featured Intermediate Blueprints

### Blueprint 1: [Real-Time Helmet & Safety Gear Detector](./safety-helmet-detector/)
- **Problem:** Automate campus and lab safety enforcement by detecting whether individuals are wearing required protective helmets.
- **Tech Stack:** Python, OpenCV, Ultralytics YOLOv8, PyTorch.
- **Architecture:**
  1. Annotated dataset of riders/workers with and without helmets.
  2. Fine-tuned YOLOv8 model for real-time bounding box prediction.
  3. OpenCV video capture stream with bounding boxes and confidence score overlay.
- **Status:** Implemented! See [safety-helmet-detector/README.md](./safety-helmet-detector/README.md).

### Blueprint 2: Multilingual Support Ticket Routing Engine
- **Problem:** Route incoming student grievance and query tickets to appropriate college departments automatically.
- **Tech Stack:** Python, Hugging Face Transformers, DistilBERT, FastAPI.
- **Architecture:**
  1. Text tokenization and multi-class classification.
  2. FastAPI `/predict` endpoint returning category and probability scores.
  3. Dockerfile for containerized deployment.

### Blueprint 3: Real-Time Hand Gesture Controller & Occlusion Robustness Benchmark
- **Problem:** Build a contact-free gesture controller (e.g. presentation slide flipper, volume controller) and benchmark tracking stability under 0%–60% synthetic partial occlusions.
- **Tech Stack:** Python, OpenCV, Google MediaPipe, NumPy, Scikit-Learn.
- **Architecture:**
  1. Real-time 21-point 3D hand landmark extraction via MediaPipe Hands.
  2. Translation-invariant normalization (anchoring relative to wrist / palm center).
  3. Synthetic partial occlusion mask simulation to evaluate confidence decay.
  4. Real-time webcam overlay and gesture action dispatcher.

---

## 🛠️ How to Add an Intermediate Project
Follow the [Standard Project Submission Template](../templates/PROJECT_TEMPLATE.md) and open a PR!
