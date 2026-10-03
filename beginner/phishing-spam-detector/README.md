# 🛡️ Phishing & Spam Message Classifier

> An introductory Natural Language Processing (NLP) project that detects whether an incoming message is legitimate (**Ham**) or fraudulent (**Phishing / Spam**) using text vectorization and machine learning.

[![AIML Club OCT](https://img.shields.io/badge/AIML_Club-OCT_Bhopal-0052CC?style=flat-square)](https://aimlcluboct.in)
[![Category](https://img.shields.io/badge/Track-Beginner_NLP-brightgreen.svg?style=flat-square)](#)
[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg?style=flat-square)](https://www.python.org/)

---

## 🎯 What You Will Learn
1. **Text Preprocessing:** Cleaning text strings, lowercasing, stripping special noise, and tokenizing URLs and numeric sequences.
2. **TF-IDF Vectorization:** Understanding how raw words are converted into mathematical feature vectors using Term Frequency-Inverse Document Frequency.
3. **Classification Algorithms:** Comparing **Multinomial Naive Bayes** (the industry gold-standard baseline for text) and **Logistic Regression**.
4. **Evaluation Metrics:** Interpreting Precision, Recall, F1-Score, and Confusion Matrix for imbalanced fraud detection.
5. **Real-Time Inference:** Predicting unseen messages with probabilistic confidence scores.

---

## 🏗️ Project Architecture

```text
Raw SMS / Email Message
         │
         ▼
[Text Preprocessing] ──► Lowercase, remove symbols, map URLs to tokens
         │
         ▼
[TF-IDF Vectorizer] ──► Converts text into numerical n-gram matrices (Unigrams + Bigrams)
         │
         ▼
[Multinomial Naive Bayes / Logistic Regression]
         │
         ▼
Prediction: 🚨 SPAM (98.4% confidence) or ✅ HAM (99.1% confidence)
```

---

## 🚀 Quickstart

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Classifier
```bash
python train_and_predict.py
```

---

## 📊 Sample Output

```text
======================================================================
🛡️  AIML Club OCT — Phishing & Spam Message Classifier Starter
======================================================================

📊 Total dataset size: 40 samples
   • Legitimate (Ham): 20 messages
   • Phishing / Spam:  20 messages

🧹 Preprocessing text messages...
   • Training samples: 28
   • Testing samples:  12

📐 Vectorizing text with TF-IDF (Unigrams + Bigrams)...
   • Vocabulary size: 218 unique n-gram features

⚙️  Training Model 1: Multinomial Naive Bayes...
   • Naive Bayes Test Accuracy: 100.00%

⚙️  Training Model 2: Logistic Regression (L2 Regularized)...
   • Logistic Regression Test Accuracy: 100.00%

======================================================================
📈 Multinomial Naive Bayes Evaluation Report:
======================================================================
                  precision    recall  f1-score   support

Ham (Legitimate)       1.00      1.00      1.00         6
 Spam (Phishing)       1.00      1.00      1.00         6

        accuracy                           1.00        12
       macro avg       1.00      1.00      1.00        12
    weighted avg       1.00      1.00      1.00        12

Confusion Matrix:
   True Negative: 6 | False Positive: 0
   False Negative: 0 | True Positive:  6

======================================================================
🔮 Real-Time Test Message Inference Demonstration:
======================================================================

Message:    "Hi, are you submitting the machine learning assignment before tomorrow's class?"
Prediction: ✅ LEGITIMATE (HAM) (Confidence: 94.20%)

Message:    "URGENT! You won 1,000,000 cash prize. Click here now to claim your money immediately!"
Prediction: 🚨 SPAM / PHISHING (Confidence: 97.85%)
```

---

## 💡 Student Challenges & Contributions

Want to contribute and upgrade this project? Here are great ways to extend it:

- [ ] **Challenge 1 (Streamlit Web App):** Create a `streamlit_app.py` with an interactive text area, confidence gauges, and highlighted keywords.
- [ ] **Challenge 2 (Kaggle Dataset Ingestion):** Add an option to train on the real Kaggle [SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset) (5,574 messages).
- [ ] **Challenge 3 (Model Serialization):** Use `joblib` or `pickle` to export the trained pipeline (`model.joblib` and `vectorizer.joblib`) for zero-latency production loading.
- [ ] **Challenge 4 (Explainability):** Use top TF-IDF weights to display *why* a message was classified as spam (e.g. key contributing words like `urgent`, `free`, `claim`).

---

## 🤝 Contributing
Read our [Contributing Guide](../../CONTRIBUTING.md) and open a pull request!
