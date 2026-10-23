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

### Blueprint 2: [Student Feedback Sentiment Analyzer](./student-feedback-sentiment-analyzer/)
- **Problem:** Analyze multi-class course and workshop feedback (Positive / Neutral / Negative) with clause-level aspect extraction and mixed-sentiment handling.
- **Tech Stack:** Python, Scikit-Learn, Pandas, NumPy, Joblib.
- **Architecture:**
  1. Multi-model benchmarking (Logistic Regression, Linear SVC, Complement Naive Bayes, Soft Voting Ensemble) with stratified cross-validation.
  2. Realistic evaluation across in-distribution, unseen phrasing, and hand-written gold test sets.
  3. Clause-level Aspect-Based Sentiment Analysis (ABSA) covering `lectures`, `labs`, `assignments`, `instructor`, `exams`, `materials`, and `pace`.
  4. Interactive CLI inference loop and model serialization (`--save` / `--load`).
- **Status:** Implemented! See [student-feedback-sentiment-analyzer/README.md](./student-feedback-sentiment-analyzer/README.md).

### Blueprint 3: Multilingual Support Ticket Routing Engine
- **Problem:** Route incoming student grievance and query tickets to appropriate college departments automatically.
- **Tech Stack:** Python, Hugging Face Transformers, DistilBERT, FastAPI.
- **Architecture:**
  1. Text tokenization and multi-class classification.
  2. FastAPI `/predict` endpoint returning category and probability scores.
  3. Dockerfile for containerized deployment.

### Blueprint 4: Real-Time Hand Gesture Controller & Occlusion Robustness Benchmark
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
