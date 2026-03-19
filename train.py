import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle
from textblob import TextBlob
import os

# --------------------------
# Load or Train Model
# --------------------------

if os.path.exists("model.pkl") and os.path.exists("vectorizer.pkl"):
    model = pickle.load(open("model.pkl", "rb"))
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
else:
    # Sample dataset
    data = {
        "review": [
            "This product is amazing",
            "Worst product ever",
            "I love this item",
            "Fake fake fake product",
            "Very good quality",
            "Do not buy this",
            "Best ever! 100% guaranteed",
            "Life-changing item, amazing!"
        ],
        "label": [0, 1, 0, 1, 0, 1, 1, 0]
    }

    df = pd.DataFrame(data)

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df["review"])

    model = LogisticRegression()
    model.fit(X, df["label"])

    # Save model
    pickle.dump(model, open("model.pkl", "wb"))
    pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

# --------------------------
# Analyze Function
# --------------------------

def analyze_review(review_text):
    X_test = vectorizer.transform([review_text])
    prob_fake = model.predict_proba(X_test)[0][1]
    result_label = "Fake" if prob_fake > 0.5 else "Genuine"

    sentiment_score = TextBlob(review_text).sentiment.polarity
    if sentiment_score > 0.1:
        sentiment = "Positive"
    elif sentiment_score < -0.1:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"

    suspicious_words_list = ["best ever", "100% guaranteed", "amazing product", "life-changing"]
    found_words = [w for w in suspicious_words_list if w in review_text.lower()]

    explanation = []
    if result_label == "Fake":
        if found_words:
            explanation.append("Uses generic/suspicious phrases")
        if sentiment == "Positive" and prob_fake > 0.8:
            explanation.append("Overly positive tone")
        if len(review_text.split()) < 3 or len(review_text.split()) > 50:
            explanation.append("Unusual review length")

    return {
        "review": review_text,
        "result": result_label,
        "confidence": float(round(prob_fake * 100, 2)),
        "sentiment": sentiment,
        "suspicious_words": found_words,
        "explanation": explanation
    }

# --------------------------
# Test Run
# --------------------------
if __name__ == "__main__":
    test_review = "Best ever! I love this product"
    print(analyze_review(test_review))
