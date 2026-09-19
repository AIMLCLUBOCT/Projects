# 🟡 Intermediate Projects

> Practical deep learning, computer vision, and NLP implementations with modular code and API endpoints.

---

## 🎯 Focus Areas
- Computer vision: Image classification, filtering, and real-time object detection using OpenCV and YOLO.
- Natural language processing: Text classification, Named Entity Recognition (NER), and sentiment analysis with Hugging Face Transformers.
- Asynchronous API serving with FastAPI and Pydantic schema validation.

---

## 📂 Featured Intermediate Blueprints

### Blueprint 1: Real-Time Helmet & Safety Gear Detector
- **Problem:** Automate campus and lab safety enforcement by detecting whether individuals are wearing required protective helmets.
- **Tech Stack:** Python, OpenCV, Ultralytics YOLOv8, PyTorch.
- **Architecture:**
  1. Annotated dataset of riders/workers with and without helmets.
  2. Fine-tuned YOLOv8 model for real-time bounding box prediction.
  3. OpenCV video capture stream with bounding boxes and confidence score overlay.

### Blueprint 2: Multilingual Support Ticket Routing Engine
- **Problem:** Route incoming student grievance and query tickets to appropriate college departments automatically.
- **Tech Stack:** Python, Hugging Face Transformers, DistilBERT, FastAPI.
- **Architecture:**
  1. Text tokenization and multi-class classification.
  2. FastAPI `/predict` endpoint returning category and probability scores.
  3. Dockerfile for containerized deployment.

---

## 🛠️ How to Add an Intermediate Project
Follow the [Standard Project Submission Template](../templates/PROJECT_TEMPLATE.md) and open a PR!
