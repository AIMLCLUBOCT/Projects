"""
Tabular Student Performance & Risk Predictor
AIML Club — Oriental College of Technology, Bhopal
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

def main():
    print("=" * 60)
    print("🎓 Student Academic Performance & Risk Predictor Starter")
    print("=" * 60)
    
    np.random.seed(42)
    n = 600
    df = pd.DataFrame({
        'attendance_pct': np.random.uniform(40, 100, n),
        'internal_marks': np.random.uniform(10, 30, n),
        'assignments_submitted': np.random.randint(2, 10, n),
        'study_hours_week': np.random.uniform(2, 25, n),
    })
    # Academic risk label (1 = At Risk / Intervention needed, 0 = On Track)
    risk_score = (100 - df['attendance_pct']) * 0.4 + (30 - df['internal_marks']) * 1.5 - df['study_hours_week'] * 1.2
    df['at_risk'] = (risk_score > 35).astype(int)

    X = df.drop(columns=['at_risk'])
    y = df['at_risk']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    print("Model Evaluation Report:")
    print(classification_report(y_test, preds, target_names=['On Track', 'At Risk']))

if __name__ == "__main__":
    main()
