import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle
from textblob import TextBlob

# --------------------------
# 1️⃣ Sample Dataset
# Replace with your full dataset later
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
    "label": [0, 1, 0, 1, 0, 1, 1, 0]  # 0 = real, 1 = fake
}
df = pd.DataFrame(data)

# --------------------------
# 2️⃣ Vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["review"])

# --------------------------
# 3️⃣ Model
model = LogisticRegression()
model.fit(X, df["label"])

# --------------------------
# 4️⃣ Function to analyze reviews
def analyze_review(review_text):
    # Vectorize
    X_test = vectorizer.transform([review_text])
    
    # Probability (confidence)
    prob_fake = model.predict_proba(X_test)[0][1]  # probability of being fake
    result_label = "Fake" if prob_fake > 0.5 else "Genuine"
    
    # Sentiment Analysis
    sentiment_score = TextBlob(review_text).sentiment.polarity
    if sentiment_score > 0.1:
        sentiment = "Positive"
    elif sentiment_score < -0.1:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"
    
    # Suspicious Words
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
    
    # Return all info as a dictionary
    return {
        "review": review_text,
        "result": result_label,
        "confidence": round(prob_fake*100, 2),
        "sentiment": sentiment,
        "suspicious_words": found_words,
        "explanation": explanation
    }

# Example usage
if __name__ == "__main__":
    test_review = "Best ever! I love this product"
    analysis = analyze_review(test_review)
    print(analysis)

# --------------------------
# 5️⃣ Save model and vectorizer
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))
print("Model and vectorizer saved!")
