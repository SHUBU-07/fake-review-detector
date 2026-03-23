import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
import os

# --------------------------
# Load CSV safely
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(BASE_DIR, "reviews.csv")

df = pd.read_csv(file_path)

# --------------------------
# CLEAN DATA
df = df.dropna(subset=["review", "label"])
df["label"] = pd.to_numeric(df["label"], errors="coerce")
df = df.dropna(subset=["label"])
df["label"] = df["label"].astype(int)
df = df[df["review"].str.strip() != ""]

print("✅ Cleaned Dataset Size:", len(df))

# --------------------------
# Features & Labels
X_text = df["review"]
y = df["label"]

# --------------------------
# Vectorization (IMPROVED)
vectorizer = TfidfVectorizer(
    ngram_range=(1,2),
    stop_words='english',
    max_features=500
)

X = vectorizer.fit_transform(X_text)

# --------------------------
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --------------------------
# MODEL (BEST FOR TEXT)
model = MultinomialNB()
model.fit(X_train, y_train)

# --------------------------
# Accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"✅ Accuracy: {accuracy*100:.2f}%")

# --------------------------
# Save model
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("✅ Model and vectorizer saved!")
