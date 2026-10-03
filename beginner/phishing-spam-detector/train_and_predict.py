"""
Phishing & Spam Message Classifier
AIML Club — Oriental College of Technology, Bhopal

A beginner-friendly Natural Language Processing (NLP) project teaching:
1. Text tokenization and TF-IDF (Term Frequency-Inverse Document Frequency) vectorization.
2. Supervised classification comparing Multinomial Naive Bayes and Logistic Regression.
3. Model evaluation with Precision, Recall, F1-Score, and Confusion Matrix.
4. Real-time inference with probability confidence scores.
"""

import math
import re
import sys
from collections import Counter, defaultdict

# Ensure UTF-8 console output on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

try:
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


def load_dataset():
    """
    Returns a representative dataset of legitimate (ham) and spam/phishing messages.
    Includes educational, academic, urgent scam, lottery, and phishing examples.
    """
    messages = [
        # Legitimate messages (Ham = 0)
        ("Hey, are we still meeting in the computer lab at 4 PM for the AI workshop?", 0),
        ("Please find attached the lecture notes for module 3 on Linear Regression.", 0),
        ("Don't forget to submit your semester project proposal by Friday midnight.", 0),
        ("The college library will remain open until 8 PM during end-semester exams.", 0),
        ("Hi team, I've pushed the latest fixes to the GitHub repository. Please review.", 0),
        ("Can you share the dataset link for the Kaggle competition we discussed?", 0),
        ("Your OTP for Oriental College student portal login is 482910. Do not share.", 0),
        ("Good morning Professor, I had a question regarding problem 4 in the assignment.", 0),
        ("Reminder: Tomorrow is the deadline for club membership renewal form.", 0),
        ("The hackathon registration is confirmed. Your team ID is OCT-AI-2026.", 0),
        ("Hey, let's grab lunch at the canteen before the afternoon data science lab.", 0),
        ("Your GitHub pull request has been approved and successfully merged to main.", 0),
        ("The bus route 4 will arrive 10 minutes late due to road maintenance.", 0),
        ("Please submit the hard copy of your fee receipt to the administrative office.", 0),
        ("Are you coming to the college coding club meeting today at the seminar hall?", 0),
        ("The exam schedule for 4th semester has been published on the official notice board.", 0),
        ("Don't forget to bring your identity card and lab manual for tomorrow's viva.", 0),
        ("Thanks for sharing the research paper on transformer architectures!", 0),
        ("Your hostel room inspection is scheduled for Wednesday 5 PM.", 0),
        ("Could you please review the draft of our paper before we submit to the conference?", 0),

        # Spam / Phishing messages (Spam = 1)
        ("URGENT: Your bank account is locked! Click http://bit.ly/secure-bank to verify immediately.", 1),
        ("Congratulations! You won a $1,000 Walmart gift card. Call 1-800-CLAIM-NOW to get your prize!", 1),
        ("Dear customer, your credit card points worth Rs 9,500 are expiring today. Redeem at http://xyz-points.cc", 1),
        ("WINNER! You have been selected for a free iPhone 15 Pro. Reply YES or visit http://win-claim.xyz", 1),
        ("Urgent notice: Your electricity power will be disconnected tonight. Call this number immediately.", 1),
        ("Get a guaranteed personal loan up to 50 Lakhs with 0% interest! No documents needed. Apply now!", 1),
        ("You have an unclaimed inheritance of 2.5 Million USD from your late relative. Send your passport details.", 1),
        ("ALERT: Suspicious login attempt detected on your PayPal. Verify your password here: http://secure-paypal-login.com", 1),
        ("Exclusive job offer! Earn Rs 5000 daily working from home just by watching YouTube videos. WhatsApp now!", 1),
        ("FINAL NOTICE: Claim your tax refund payment of $850. Submit your banking details before 5 PM.", 1),
        ("Hot deal! Lose 15 kgs in 7 days without diet or exercise. Order magic pills now at 80% discount!", 1),
        ("Your package delivery is pending due to unpaid customs tax of Rs 49. Pay here: http://track-parcel-fees.top", 1),
        ("Double your Bitcoin in 24 hours! Guaranteed 200% return on automated crypto trading bot.", 1),
        ("Free recharge of Rs 499 for all students! Click the link to activate your complimentary mobile pack.", 1),
        ("You have won lottery ticket #94827 worth 1 Crore rupees. Transfer registration charges of Rs 2500 to claim.", 1),
        ("Warning: Your Netflix subscription will expire today. Update billing card details at http://netflix-billing.icu", 1),
        ("Congratulations! Selected for paid internship in Google with 1 Lakh stipend without interview. Click to pay fee.", 1),
        ("URGENT: Your KYC is suspended by SBI. Click http://sbi-kyc-update.online to avoid total account freeze.", 1),
        ("Work 2 hours daily from your mobile phone and earn $200 daily. Limited spots available, register now!", 1),
        ("Important security update: Download and install this security patch immediately from http://antivirus-patch.ru", 1),
    ]

    return messages


def preprocess_text(text: str) -> str:
    """
    Cleans text by lowercasing, stripping special symbols/URLs, and normalizing whitespace.
    """
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " url_link ", text)
    text = re.sub(r"\d+", " num_token ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def run_pure_python_pipeline(raw_messages):
    """
    Educational reference pipeline demonstrating TF-IDF & Naive Bayes mechanics
    in pure standard library Python without requiring external wheels.
    """
    print("\n💡 Running Educational Built-in Pipeline (Zero External Dependencies):")
    cleaned_messages = [(preprocess_text(txt), label) for txt, label in raw_messages]

    # Stratified split: alternate even/odd indices for balanced train/test
    train_data = [msg for i, msg in enumerate(cleaned_messages) if i % 3 != 0]
    test_data = [msg for i, msg in enumerate(cleaned_messages) if i % 3 == 0]

    # Calculate Term Frequencies & Document Frequencies
    doc_freq = defaultdict(int)
    for text, _ in train_data:
        words = set(text.split())
        for w in words:
            doc_freq[w] += 1

    total_docs = len(train_data)
    vocab = {word: idx for idx, word in enumerate(doc_freq.keys())}
    idf = {word: math.log((total_docs + 1) / (df + 1)) + 1 for word, df in doc_freq.items()}

    def get_tfidf_vec(text):
        words = text.split()
        tf = Counter(words)
        total_words = len(words) or 1
        vec = defaultdict(float)
        for w, count in tf.items():
            if w in vocab:
                vec[w] = (count / total_words) * idf[w]
        return vec

    # Train class priors and feature likelihoods
    spam_docs = [get_tfidf_vec(txt) for txt, lbl in train_data if lbl == 1]
    ham_docs = [get_tfidf_vec(txt) for txt, lbl in train_data if lbl == 0]

    p_spam = len(spam_docs) / len(train_data)
    p_ham = len(ham_docs) / len(train_data)

    def calc_feature_probs(docs_list):
        total_weights = defaultdict(float)
        sum_total = 0.0
        for doc in docs_list:
            for w, weight in doc.items():
                total_weights[w] += weight
                sum_total += weight
        # Laplace smoothing
        vocab_len = len(vocab)
        return {w: (total_weights[w] + 1.0) / (sum_total + vocab_len) for w in vocab}, sum_total + vocab_len

    spam_probs, _ = calc_feature_probs(spam_docs)
    ham_probs, _ = calc_feature_probs(ham_docs)

    def predict(text):
        vec = get_tfidf_vec(text)
        log_prob_spam = math.log(p_spam)
        log_prob_ham = math.log(p_ham)

        for w in vec:
            if w in vocab:
                log_prob_spam += vec[w] * math.log(spam_probs.get(w, 1e-6))
                log_prob_ham += vec[w] * math.log(ham_probs.get(w, 1e-6))

        # Softmax normalization for display
        max_log = max(log_prob_spam, log_prob_ham)
        exp_spam = math.exp(log_prob_spam - max_log)
        exp_ham = math.exp(log_prob_ham - max_log)
        prob_spam = exp_spam / (exp_spam + exp_ham)
        return 1 if log_prob_spam > log_prob_ham else 0, prob_spam

    correct = 0
    for txt, true_lbl in test_data:
        pred_lbl, _ = predict(txt)
        if pred_lbl == true_lbl:
            correct += 1

    acc = correct / len(test_data)
    print(f"   • Built-in Baseline Validation Accuracy: {acc * 100:.1f}%")

    sample_tests = [
        "Hi, are you submitting the machine learning assignment before tomorrow's class?",
        "URGENT! You won 1,000,000 cash prize. Click here now to claim your money immediately!",
        "Can we reschedule our study group session to 6 PM in the OCT college library?",
        "Dear user, your bank debit card is blocked. Click http://verify-sbi-card.info to unblock."
    ]

    print("\n🔮 Interactive Demonstration Predictions:")
    for msg in sample_tests:
        cleaned = preprocess_text(msg)
        lbl, prob_spam = predict(cleaned)
        conf = prob_spam if lbl == 1 else (1.0 - prob_spam)
        tag = "🚨 SPAM / PHISHING" if lbl == 1 else "✅ LEGITIMATE (HAM)"
        print(f"\nMessage:    \"{msg}\"")
        print(f"Prediction: {tag} (Confidence: {conf * 100:.2f}%)")


def run_sklearn_pipeline(raw_messages):
    """
    Standard production-grade scikit-learn pipeline using TfidfVectorizer,
    MultinomialNB, and LogisticRegression.
    """
    df = pd.DataFrame(raw_messages, columns=["text", "label"])
    print(f"\n📊 Total dataset size: {len(df)} samples")
    print(f"   • Legitimate (Ham): {(df['label'] == 0).sum()} messages")
    print(f"   • Phishing / Spam:  {(df['label'] == 1).sum()} messages")

    # Preprocessing
    df["cleaned_text"] = df["text"].apply(preprocess_text)

    # Train / Test split
    X_train, X_test, y_train, y_test = train_test_split(
        df["cleaned_text"],
        df["label"],
        test_size=0.3,
        random_state=42,
        stratify=df["label"]
    )
    print(f"   • Training samples: {len(X_train)}")
    print(f"   • Testing samples:  {len(X_test)}")

    # Vectorization
    print("\n📐 Vectorizing text with TF-IDF (Unigrams + Bigrams)...")
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        stop_words="english",
        min_df=1
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    print(f"   • Vocabulary size: {len(vectorizer.get_feature_names_out())} unique n-gram features")

    # Multinomial Naive Bayes
    print("\n⚙️  Training Model 1: Multinomial Naive Bayes...")
    nb_model = MultinomialNB(alpha=0.5)
    nb_model.fit(X_train_vec, y_train)
    nb_preds = nb_model.predict(X_test_vec)
    nb_acc = accuracy_score(y_test, nb_preds)
    print(f"   • Naive Bayes Test Accuracy: {nb_acc * 100:.2f}%")

    # Logistic Regression
    print("\n⚙️  Training Model 2: Logistic Regression (L2 Regularized)...")
    lr_model = LogisticRegression(C=1.0, random_state=42)
    lr_model.fit(X_train_vec, y_train)
    lr_preds = lr_model.predict(X_test_vec)
    lr_acc = accuracy_score(y_test, lr_preds)
    print(f"   • Logistic Regression Test Accuracy: {lr_acc * 100:.2f}%")

    # Evaluation
    print("\n" + "=" * 70)
    print("📈 Multinomial Naive Bayes Evaluation Report:")
    print("=" * 70)
    print(classification_report(y_test, nb_preds, target_names=["Ham (Legitimate)", "Spam (Phishing)"]))

    cm = confusion_matrix(y_test, nb_preds)
    print("Confusion Matrix:")
    print(f"   True Negative: {cm[0][0]} | False Positive: {cm[0][1]}")
    print(f"   False Negative: {cm[1][0]} | True Positive:  {cm[1][1]}")

    sample_tests = [
        "Hi, are you submitting the machine learning assignment before tomorrow's class?",
        "URGENT! You won 1,000,000 cash prize. Click here now to claim your money immediately!",
        "Can we reschedule our study group session to 6 PM in the OCT college library?",
        "Dear user, your bank debit card is blocked. Click http://verify-sbi-card.info to unblock."
    ]

    print("\n" + "=" * 70)
    print("🔮 Real-Time Test Message Inference Demonstration:")
    print("=" * 70)

    for msg in sample_tests:
        cleaned = preprocess_text(msg)
        vec = vectorizer.transform([cleaned])
        pred_label = nb_model.predict(vec)[0]
        prob = nb_model.predict_proba(vec)[0]

        status = "🚨 SPAM / PHISHING" if pred_label == 1 else "✅ LEGITIMATE (HAM)"
        confidence = prob[pred_label] * 100

        print(f"\nMessage:    \"{msg}\"")
        print(f"Prediction: {status} (Confidence: {confidence:.2f}%)")


def main():
    print("=" * 70)
    print("🛡️  AIML Club OCT — Phishing & Spam Message Classifier Starter")
    print("=" * 70)

    raw_messages = load_dataset()

    if SKLEARN_AVAILABLE:
        run_sklearn_pipeline(raw_messages)
    else:
        print("\nℹ️  Notice: scikit-learn is not installed in the current environment.")
        print("   To enable full sklearn pipelines: pip install -r requirements.txt")
        run_pure_python_pipeline(raw_messages)

    print("\n" + "=" * 70)
    print("🎉 Project executed successfully! Ready for student extensions.")
    print("=" * 70)


if __name__ == "__main__":
    main()
