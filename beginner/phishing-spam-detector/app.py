"""
Phishing & Spam Message Classifier — Interactive Streamlit UI
AIML Club — Oriental College of Technology, Bhopal
"""

import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

from train_and_predict import load_dataset, preprocess_text

st.set_page_config(page_title="Spam & Phishing Detector", page_icon="🛡️")


@st.cache_resource
def train_model():
    """Train the TF-IDF + Naive Bayes model once and cache it."""
    raw_messages = load_dataset()
    texts = [preprocess_text(msg) for msg, _ in raw_messages]
    labels = [label for _, label in raw_messages]

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", min_df=1)
    X = vectorizer.fit_transform(texts)

    model = MultinomialNB(alpha=0.5)
    model.fit(X, labels)

    return vectorizer, model


vectorizer, model = train_model()

st.title("🛡️ SMS and Email Spam / Phishing Detector")
st.subheader("Paste a message below to check if it's safe or suspicious")

col1, col2 = st.columns(2)
sample_scam = "URGENT: Your bank account is locked! Click http://bit.ly/secure-bank to verify immediately."
sample_safe = "Hey, are we still meeting in the computer lab at 4 PM for the AI workshop?"

if "message_text" not in st.session_state:
    st.session_state.message_text = ""

with col1:
    if st.button("Test Scam Alert"):
        st.session_state.message_text = sample_scam

with col2:
    if st.button("Test College Club Message"):
        st.session_state.message_text = sample_safe

message = st.text_area(
    "Message text",
    value=st.session_state.message_text,
    height=150,
    placeholder="Paste an SMS, email, or message here...",
)

if st.button("Analyze Message", type="primary"):
    if not message.strip():
        st.warning("Please enter a message to analyze.")
    else:
        cleaned = preprocess_text(message)
        vec = vectorizer.transform([cleaned])
        prediction = model.predict(vec)[0]
        probabilities = model.predict_proba(vec)[0]
        confidence = probabilities[prediction]

        if prediction == 1:
            st.error("🚨 This looks like SPAM / PHISHING")
        else:
            st.success("✅ This looks like a LEGITIMATE message")

        st.metric("Confidence", f"{confidence * 100:.1f}%")
        st.progress(float(confidence))
