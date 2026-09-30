# 🎓 Student Academic Performance & Risk Predictor

> A machine learning pipeline to identify students needing academic intervention prior to semester exams.

[![Status: Blueprint Starter](https://img.shields.io/badge/Status-Starter_Blueprint-brightgreen.svg?style=flat-square)](#)
[![Domain: Tabular ML](https://img.shields.io/badge/Domain-Tabular_ML-blue.svg?style=flat-square)](#)
[![Contributions Welcome](https://img.shields.io/badge/Contributions-Welcome-orange.svg?style=flat-square)](../../CONTRIBUTING.md)

---

## 🎯 Problem Statement
Early identification of students struggling with coursework is critical for timely mentoring and academic support before university end-semester examinations.

## 💡 Solution Architecture
- **Data Features:** Attendance percentage, internal mid-semester assessments, assignment submission rates, and weekly self-study logs.
- **Model:** Random Forest Classifier & Gradient Boosting (scikit-learn).
- **Target:** Academic Risk Flag (`0` = On Track, `1` = At Risk).

## 🚀 Quickstart
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate

pip install -r requirements.txt
python train_and_predict.py
```

## 🤝 Open Contributions (Good First Issues)
- [ ] Add a Streamlit web UI to input student metrics interactively.
- [ ] Implement SHAP or Feature Importance visualization.
- [ ] Ingest real anonymized benchmark datasets.
