import streamlit as st
import pickle
from textblob import TextBlob
import matplotlib.pyplot as plt
import datetime
import time

# --------------------------
# Load model & vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# --------------------------
# Title
st.title("🕵️ Fake Review Detector AI")
st.write("Analyze whether a review is Fake or Genuine with AI insights")

# --------------------------
# Input
review = st.text_area("Enter your review")

# --------------------------
# Session history
if "history" not in st.session_state:
    st.session_state.history = []

# --------------------------
# Analyze Button
if st.button("Analyze Review"):

    if review.strip() == "":
        st.warning("Please enter a review!")
    else:
        # Loading animation
        with st.spinner("🤖 AI is analyzing..."):
            time.sleep(1)

        # --------------------------
        # Prediction
        X_test = vectorizer.transform([review])
        prob_fake = model.predict_proba(X_test)[0][1]
        result = "Fake" if prob_fake > 0.5 else "Genuine"

        # --------------------------
        # Trust Score
        trust_score = round((1 - prob_fake) * 100, 2)

        # --------------------------
        # Sentiment
        sentiment_score = TextBlob(review).sentiment.polarity
        if sentiment_score > 0.1:
            sentiment = "Positive"
        elif sentiment_score < -0.1:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"

        # --------------------------
        # Suspicious words
        suspicious_words_list = [
            "best ever", "100% guaranteed",
            "amazing product", "life-changing"
        ]
        found_words = [w for w in suspicious_words_list if w in review.lower()]

        # --------------------------
        # Explanation logic
        explanation = []

        if result == "Fake":
            if found_words:
                explanation.append("Uses generic/suspicious phrases")
            if sentiment == "Positive" and prob_fake > 0.8:
                explanation.append("Overly positive tone")
            if len(review.split()) < 3 or len(review.split()) > 50:
                explanation.append("Unusual review length")

        # Pattern detection
        words = review.lower().split()
        if len(words) != len(set(words)):
            explanation.append("Repeated words detected")

        if review.count("!") > 3:
            explanation.append("Too many exclamation marks")

        # --------------------------
        # DISPLAY RESULTS

        st.subheader("📊 Detailed Analysis")

        if result == "Fake":
            st.error(f"⚠️ Fake Review ({round(prob_fake*100,2)}%)")
        else:
            st.success(f"✅ Genuine Review ({round(prob_fake*100,2)}%)")

        st.write("**Review:**", review)
        st.write("**Sentiment:**", sentiment)

        # Explanation
        st.subheader("🤖 Why this review is fake?")
        if explanation:
            for e in explanation:
                st.write("•", e)
        else:
            st.write("No major issues detected ✅")

        # Suspicious words
        if found_words:
            st.write("**⚠️ Suspicious Words:**", found_words)

        # Trust Score
        st.subheader("📈 Product Trust Score")
        st.write(f"{trust_score}%")

        # --------------------------
        # VISUALIZATION

        st.subheader("📊 Visual Insights")

        # Pie Chart
        fig, ax = plt.subplots()
        ax.pie([prob_fake, 1-prob_fake],
               labels=["Fake", "Genuine"],
               autopct="%1.1f%%")
        st.pyplot(fig)

        # Sentiment Bar
        fig2, ax2 = plt.subplots()
        ax2.bar(["Sentiment Score"], [sentiment_score])
        st.pyplot(fig2)

        # --------------------------
        # Save History
        st.session_state.history.append({
            "review": review,
            "result": result,
            "time": datetime.datetime.now().strftime("%H:%M:%S")
        })

# --------------------------
# SHOW HISTORY

st.subheader("🕒 Review History")

for item in st.session_state.history:
    st.write(f"{item['time']} - {item['review']} → {item['result']}")