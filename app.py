import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import json
import datetime
import requests
from bs4 import BeautifulSoup

# --------------------------
# Load model
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# --------------------------
# PREMIUM UI CSS
st.markdown("""
<style>
body {
    background-color: #0E1117;
    color: white;
}
.stButton>button {
    background-color: #ff4b4b;
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
}
.stTextArea textarea {
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# --------------------------
# TITLE
st.markdown("# 🚀 AI Fake Review Detector")
st.markdown("### ⚡ Analyze reviews, detect fake content, and generate trust score")

# --------------------------
# SIMPLE SENTIMENT (no TextBlob)
def get_sentiment(text):
    positive_words = ["good", "great", "amazing", "love", "excellent"]
    negative_words = ["bad", "worst", "hate", "terrible"]

    if any(word in text.lower() for word in positive_words):
        return "Positive"
    elif any(word in text.lower() for word in negative_words):
        return "Negative"
    else:
        return "Neutral"

# --------------------------
# ANALYZE FUNCTION
def analyze_review(review_text):
    X_test = vectorizer.transform([review_text])
    prob_fake = model.predict_proba(X_test)[0][1]
    result_label = "Fake" if prob_fake > 0.5 else "Genuine"

    sentiment = get_sentiment(review_text)

    suspicious_words = ["best ever", "100% guaranteed", "life-changing"]
    found_words = [w for w in suspicious_words if w in review_text.lower()]

    explanation = []
    if result_label == "Fake":
        if found_words:
            explanation.append("Uses suspicious marketing phrases")
        if sentiment == "Positive" and prob_fake > 0.7:
            explanation.append("Overly positive tone")
        if len(review_text.split()) < 3 or len(review_text.split()) > 40:
            explanation.append("Unusual review length")

    return prob_fake, result_label, sentiment, found_words, explanation

# --------------------------
# HISTORY SAVE
def save_history(review, result, confidence, sentiment):
    file = "history.json"

    record = {
        "review": review,
        "result": result,
        "confidence": confidence,
        "sentiment": sentiment,
        "time": str(datetime.datetime.now())
    }

    try:
        with open(file, "r") as f:
            data = json.load(f)
    except:
        data = []

    data.append(record)

    with open(file, "w") as f:
        json.dump(data, f, indent=4)

# --------------------------
# URL REVIEW EXTRACTION
def extract_reviews(url):
    reviews = []
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        res = requests.get(url, headers=headers)
        soup = BeautifulSoup(res.content, "html.parser")

        for p in soup.find_all("p"):
            text = p.get_text().strip()
            if len(text) > 30:
                reviews.append(text)

    except:
        st.error("Error fetching reviews")

    return reviews[:10]

# --------------------------
# SIDEBAR OPTIONS
option = st.sidebar.selectbox("Choose Input", ["Text", "File Upload", "URL Analyzer"])

review_text = ""

# TEXT INPUT
if option == "Text":
    review_text = st.text_area("Enter Review")

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

        genuine = sum(result_df["result"] == "Genuine")
        trust_score = (genuine / len(result_df)) * 100
        st.subheader(f"🔥 Product Trust Score: {trust_score:.2f}%")

# URL ANALYZER
elif option == "URL Analyzer":
    url = st.text_input("Paste Product URL")

    if st.button("Analyze URL"):
        reviews = extract_reviews(url)

        if reviews:
            st.write(reviews)

            results = []
            for r in reviews:
                prob_fake, result_label, sentiment, _, _ = analyze_review(r)
                results.append({
                    "review": r,
                    "result": result_label,
                    "confidence": prob_fake * 100,
                    "sentiment": sentiment
                })

            df = pd.DataFrame(results)
            st.write(df)

            genuine = sum(df["result"] == "Genuine")
            trust_score = (genuine / len(df)) * 100
            st.subheader(f"🔥 Product Trust Score: {trust_score:.2f}%")

# --------------------------
# ANALYZE BUTTON
if st.button("Analyze Review") and review_text:
    with st.spinner("🤖 AI analyzing..."):

        prob_fake, result_label, sentiment, words, explanation = analyze_review(review_text)

        if result_label == "Fake":
            st.error(f"❌ Fake Review ({prob_fake*100:.2f}%)")
        else:
            st.success(f"✅ Genuine Review ({prob_fake*100:.2f}%)")

        st.progress(prob_fake)
        st.write(f"Sentiment: {sentiment}")

        if words:
            st.write("⚠ Suspicious Words:", ", ".join(words))

        st.subheader("Why this review?")
        for e in explanation:
            st.write("-", e)

        save_history(review_text, result_label, prob_fake*100, sentiment)

        # GRAPH
        col1, col2 = st.columns(2)

        with col1:
            plt.figure()
            plt.pie([prob_fake, 1-prob_fake], labels=["Fake","Genuine"], autopct='%1.1f%%')
            st.pyplot(plt)

        with col2:
            plt.figure()
            vals = [
                1 if sentiment=="Positive" else 0,
                1 if sentiment=="Neutral" else 0,
                1 if sentiment=="Negative" else 0
            ]
            plt.bar(["Positive","Neutral","Negative"], vals)
            st.pyplot(plt)

# --------------------------
# HISTORY DISPLAY
st.sidebar.subheader("📁 History")
try:
    with open("history.json", "r") as f:
        history = json.load(f)
        for item in reversed(history[-5:]):
            st.sidebar.write(f"{item['result']} ({item['confidence']:.1f}%)")
except:
    st.sidebar.write("No history")
