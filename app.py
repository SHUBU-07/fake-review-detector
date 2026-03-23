import streamlit as st
import pickle
import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt
import json
import datetime
import speech_recognition as sr

# --------------------------
# Load model
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# --------------------------
# Page config
st.set_page_config(page_title="Fake Review Detector", layout="wide")
st.title("🚀 AI Fake Review Detector - Final Boss Mode")

# --------------------------
# Analyze Function
def analyze_review(review_text):
    X_test = vectorizer.transform([review_text])
    prob_fake = model.predict_proba(X_test)[0][1]
    result_label = "Fake" if prob_fake > 0.5 else "Genuine"

    # Sentiment
    sentiment_score = TextBlob(review_text).sentiment.polarity
    if sentiment_score > 0.1:
        sentiment = "Positive"
    elif sentiment_score < -0.1:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"

    # Suspicious words
    suspicious_words_list = ["best ever", "100% guaranteed", "amazing product", "life-changing"]
    found_words = [w for w in suspicious_words_list if w in review_text.lower()]

    # Explanation
    explanation = []
    if result_label == "Fake":
        if found_words:
            explanation.append("Uses generic/suspicious phrases")
        if sentiment == "Positive" and prob_fake > 0.8:
            explanation.append("Overly positive tone")
        if len(review_text.split()) < 3 or len(review_text.split()) > 50:
            explanation.append("Unusual review length")

    return prob_fake, result_label, sentiment, found_words, explanation

# --------------------------
# Save History
def save_history(review, result, confidence, sentiment):
    history_file = "history.json"

    record = {
        "review": review,
        "result": result,
        "confidence": confidence,
        "sentiment": sentiment,
        "time": str(datetime.datetime.now())
    }

    try:
        with open(history_file, "r") as f:
            data = json.load(f)
    except:
        data = []

    data.append(record)

    with open(history_file, "w") as f:
        json.dump(data, f, indent=4)

# --------------------------
# INPUT OPTIONS
option = st.sidebar.selectbox("Choose Input Type", ["Text", "Voice", "File Upload"])

review_text = ""

# TEXT INPUT
if option == "Text":
    review_text = st.text_area("Enter Review")

# VOICE INPUT
elif option == "Voice":
    if st.button("🎙 Speak"):
        r = sr.Recognizer()
        with sr.Microphone() as source:
            st.info("Speak now...")
            audio = r.listen(source)
            try:
                review_text = r.recognize_google(audio)
                st.success(f"You said: {review_text}")
            except:
                st.error("Could not understand audio")

# FILE UPLOAD
elif option == "File Upload":
    file = st.file_uploader("Upload CSV with 'review' column")
    if file:
        df = pd.read_csv(file)
        st.write(df)

        results = []
        for review in df["review"]:
            prob_fake, result_label, sentiment, _, _ = analyze_review(review)
            results.append({
                "review": review,
                "result": result_label,
                "confidence": prob_fake * 100,
                "sentiment": sentiment
            })

        result_df = pd.DataFrame(results)
        st.write(result_df)

        # Trust Score
        genuine_count = sum(result_df["result"] == "Genuine")
        trust_score = (genuine_count / len(result_df)) * 100
        st.subheader(f"📊 Product Trust Score: {trust_score:.2f}%")

# --------------------------
# ANALYZE BUTTON
if st.button("Analyze Review") and review_text:
    prob_fake, result_label, sentiment, found_words, explanation = analyze_review(review_text)

    st.subheader(f"Result: {result_label}")
    st.write(f"Confidence: {prob_fake*100:.2f}%")
    st.write(f"Sentiment: {sentiment}")

    if found_words:
        st.write("⚠ Suspicious Words:", ", ".join(found_words))

    # Explanation
    st.subheader("🤖 Why this review?")
    for e in explanation:
        st.write("-", e)

    # Save history
    save_history(review_text, result_label, prob_fake*100, sentiment)

    # --------------------------
    # GRAPH
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Fake vs Genuine")
        plt.figure()
        plt.pie([prob_fake, 1-prob_fake], labels=["Fake", "Genuine"], autopct='%1.1f%%')
        st.pyplot(plt)

    with col2:
        st.subheader("Sentiment")
        values = [
            1 if sentiment=="Positive" else 0,
            1 if sentiment=="Neutral" else 0,
            1 if sentiment=="Negative" else 0
        ]
        plt.figure()
        plt.bar(["Positive", "Neutral", "Negative"], values)
        st.pyplot(plt)

# --------------------------
# HISTORY
st.sidebar.subheader("📁 Review History")
try:
    with open("history.json", "r") as f:
        history = json.load(f)
        for item in reversed(history[-5:]):
            st.sidebar.write(f"{item['result']} ({item['confidence']:.1f}%)")
except:
    st.sidebar.write("No history yet")
