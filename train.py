import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

# --------------------------
# Dataset (you can expand later)
data = {
    "review": [
        "This product is amazing",
        "Worst product ever",
        "I love this item",
        "Fake fake fake product",
        "Very good quality",
        "Do not buy this",
        "Best ever! 100% guaranteed",
        "Amazing product highly recommended",
        "Terrible experience waste of money",
        "Good and reliable product"
    ],
    "label": [0,1,0,1,0,1,1,1,1,0]  # 0 = genuine, 1 = fake
}

df = pd.DataFrame(data)

# --------------------------
# Vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["review"])

# --------------------------
# Model
model = LogisticRegression()
model.fit(X, df["label"])

# --------------------------
# Save files
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("✅ Model trained and saved!")
