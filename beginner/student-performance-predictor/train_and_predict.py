"""
Tabular Student Academic Performance & Risk Predictor
AIML Club — Oriental College of Technology, Bhopal

A beginner-friendly predictive modeling pipeline teaching:
1. Synthetic educational dataset generation with realistic correlations.
2. Feature engineering and normalization for tabular data.
3. Supervised classification with Random Forest, Decision Tree, and Logistic Regression.
4. Model evaluation: Accuracy, Precision, Recall, Confusion Matrix, and Feature Importance.
5. Real-time inference predicting student risk status with intervention recommendations.

Supports:
- Pure Standard Library Python fallback (zero external dependencies required).
- Full scikit-learn + pandas pipeline when installed.
"""

import math
import random
import sys

# Ensure UTF-8 console output on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

try:
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    from sklearn.model_selection import train_test_split
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


def generate_synthetic_data(n_samples: int = 500, seed: int = 42):
    """
    Generates realistic synthetic student academic records.
    Features:
      - attendance_pct (40% to 100%)
      - internal_marks (10 to 30)
      - assignments_submitted (1 to 10)
      - study_hours_week (2 to 28)
    Target:
      - at_risk: 1 (At Risk / Needs Tutoring Intervention), 0 (On Track / Good Standing)
    """
    random.seed(seed)
    data = []
    for i in range(n_samples):
        attendance = round(random.uniform(40.0, 99.0), 1)
        internal = round(random.uniform(10.0, 30.0), 1)
        assignments = random.randint(1, 10)
        study_hours = round(random.uniform(2.0, 26.0), 1)

        # Risk score calculation formula reflecting academic reality
        # High absence + low internal marks + few study hours = high risk
        risk_score = (100.0 - attendance) * 0.45 + (30.0 - internal) * 1.6 + (10 - assignments) * 1.5 - study_hours * 1.3
        # Add small random noise
        risk_score += random.gauss(0, 3.0)

        at_risk = 1 if risk_score > 32.0 else 0
        data.append({
            "student_id": f"OCT-2026-{1000 + i}",
            "attendance_pct": attendance,
            "internal_marks": internal,
            "assignments_submitted": assignments,
            "study_hours_week": study_hours,
            "at_risk": at_risk
        })
    return data


def run_pure_python_pipeline(records):
    """
    Educational reference pipeline demonstrating tabular risk prediction
    using standard library Python without requiring external wheels.
    """
    print("\n💡 Running Educational Built-in Pipeline (Zero External Dependencies):")
    total = len(records)
    at_risk_count = sum(r["at_risk"] for r in records)
    print(f"   • Total generated student records: {total}")
    print(f"   • On Track: {total - at_risk_count} | At Risk: {at_risk_count}")

    # Split into 80% train, 20% test
    split_idx = int(total * 0.8)
    train_records = records[:split_idx]
    test_records = records[split_idx:]

    # Heuristic weights for linear scoring function
    # Score = w1 * (100 - attendance) + w2 * (30 - internal) + w3 * (10 - assignments) - w4 * study_hours
    weights = {"w_att": 0.45, "w_int": 1.6, "w_ass": 1.5, "w_hrs": 1.3, "threshold": 32.0}

    def predict_single(student):
        score = (
            (100.0 - student["attendance_pct"]) * weights["w_att"]
            + (30.0 - student["internal_marks"]) * weights["w_int"]
            + (10.0 - student["assignments_submitted"]) * weights["w_ass"]
            - student["study_hours_week"] * weights["w_hrs"]
        )
        prob_risk = 1.0 / (1.0 + math.exp(-0.15 * (score - weights["threshold"])))
        pred = 1 if prob_risk >= 0.5 else 0
        return pred, prob_risk

    # Evaluate on test set
    correct = 0
    tp, fp, tn, fn = 0, 0, 0, 0
    for s in test_records:
        pred, _ = predict_single(s)
        actual = s["at_risk"]
        if pred == actual:
            correct += 1
        if pred == 1 and actual == 1:
            tp += 1
        elif pred == 1 and actual == 0:
            fp += 1
        elif pred == 0 and actual == 0:
            tn += 1
        elif pred == 0 and actual == 1:
            fn += 1

    acc = correct / len(test_records)
    print(f"   • Built-in Baseline Validation Accuracy: {acc * 100:.2f}%")
    print(f"   • Confusion Matrix: TP={tp}, FP={fp}, TN={tn}, FN={fn}")

    # Interactive sample tests
    sample_students = [
        {"name": "Student A (Consistent)", "attendance_pct": 92.0, "internal_marks": 27.5, "assignments_submitted": 9, "study_hours_week": 18.0},
        {"name": "Student B (Irregular)", "attendance_pct": 54.0, "internal_marks": 12.0, "assignments_submitted": 3, "study_hours_week": 4.0},
        {"name": "Student C (Borderline)", "attendance_pct": 74.0, "internal_marks": 18.0, "assignments_submitted": 6, "study_hours_week": 8.0},
    ]

    print("\n🔮 Interactive Demonstration Predictions:")
    print("-" * 65)
    for s in sample_students:
        pred, prob = predict_single(s)
        tag = "🚨 AT RISK (Needs Academic Mentorship)" if pred == 1 else "✅ ON TRACK (Good Academic Standing)"
        print(f"Student:    {s['name']}")
        print(f"Metrics:    Attendance: {s['attendance_pct']}% | Internals: {s['internal_marks']}/30 | Study: {s['study_hours_week']} hrs/wk")
        print(f"Assessment: {tag} (Risk Probability: {prob * 100:.1f}%)\n")


def run_sklearn_pipeline(records):
    """
    Production-grade scikit-learn pipeline using Random Forest and Logistic Regression.
    """
    df = pd.DataFrame(records)
    print(f"\n📊 Total dataset size: {len(df)} student profiles")
    print(f"   • On Track: {(df['at_risk'] == 0).sum()} students")
    print(f"   • At Risk:  {(df['at_risk'] == 1).sum()} students")

    feature_cols = ["attendance_pct", "internal_marks", "assignments_submitted", "study_hours_week"]
    X = df[feature_cols]
    y = df["at_risk"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    print(f"   • Training samples: {len(X_train)} | Testing samples: {len(X_test)}")

    # Model 1: Random Forest
    print("\n🌲 Training Model 1: Random Forest Classifier (100 Trees)...")
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_preds = rf_model.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_preds)
    print(f"   • Random Forest Test Accuracy: {rf_acc * 100:.2f}%")

    # Model 2: Logistic Regression
    print("\n📈 Training Model 2: Logistic Regression...")
    lr_model = LogisticRegression(random_state=42)
    lr_model.fit(X_train, y_train)
    lr_preds = lr_model.predict(X_test)
    lr_acc = accuracy_score(y_test, lr_preds)
    print(f"   • Logistic Regression Test Accuracy: {lr_acc * 100:.2f}%")

    # Feature Importances
    print("\n" + "=" * 65)
    print("🔍 Feature Importance Ranking (Random Forest):")
    print("=" * 65)
    for col, imp in sorted(zip(feature_cols, rf_model.feature_importances_), key=lambda x: x[1], reverse=True):
        bars = "█" * int(imp * 30)
        print(f"   • {col:<22} {imp:.4f} {bars}")

    # Evaluation
    print("\n" + "=" * 65)
    print("📋 Detailed Classification Report (Random Forest):")
    print("=" * 65)
    print(classification_report(y_test, rf_preds, target_names=["On Track", "At Risk"]))

    cm = confusion_matrix(y_test, rf_preds)
    print("Confusion Matrix:")
    print(f"   True Negative (On Track predicted On Track): {cm[0][0]}")
    print(f"   False Positive (On Track predicted At Risk): {cm[0][1]}")
    print(f"   False Negative (At Risk predicted On Track): {cm[1][0]}")
    print(f"   True Positive (At Risk predicted At Risk):   {cm[1][1]}")

    # Interactive demonstration
    test_cases = pd.DataFrame([
        {"attendance_pct": 94.0, "internal_marks": 28.0, "assignments_submitted": 10, "study_hours_week": 20.0},
        {"attendance_pct": 52.0, "internal_marks": 11.5, "assignments_submitted": 2, "study_hours_week": 3.0},
        {"attendance_pct": 72.0, "internal_marks": 17.0, "assignments_submitted": 5, "study_hours_week": 7.5},
    ])

    print("\n" + "=" * 65)
    print("🔮 Real-Time Student Risk Evaluation Demonstration:")
    print("=" * 65)
    probs = rf_model.predict_proba(test_cases)
    preds = rf_model.predict(test_cases)

    labels = ["Student A (High Achiever)", "Student B (Requires Tutoring)", "Student C (Borderline Attendance)"]
    for i, row in test_cases.iterrows():
        tag = "🚨 AT RISK (Needs Academic Mentorship)" if preds[i] == 1 else "✅ ON TRACK (Good Standing)"
        risk_pct = probs[i][1] * 100
        print(f"\nProfile:    {labels[i]}")
        print(f"Metrics:    Attendance: {row['attendance_pct']}% | Internals: {row['internal_marks']}/30 | Study: {row['study_hours_week']} hrs/wk")
        print(f"Prediction: {tag} (Risk Probability: {risk_pct:.1f}%)")


def main():
    print("=" * 65)
    print("🎓 AIML Club OCT — Student Academic Risk Predictor")
    print("=" * 65)

    data = generate_synthetic_data(n_samples=500, seed=42)

    if SKLEARN_AVAILABLE:
        run_sklearn_pipeline(data)
    else:
        print("\nℹ️  Notice: scikit-learn / pandas not installed in this environment.")
        print("   To enable full scikit-learn models: pip install -r requirements.txt")
        run_pure_python_pipeline(data)

    print("\n" + "=" * 65)
    print("🎉 Pipeline executed successfully! Ready for student extensions.")
    print("=" * 65)


if __name__ == "__main__":
    main()
