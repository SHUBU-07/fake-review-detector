# 🕵️ Fake Review Detector AI

An AI-powered web application that detects whether a product review is **Fake or Genuine** using Machine Learning and Natural Language Processing (NLP).

---

## 🚀 Features

* Fake / Genuine Review Detection
* Confidence Score (%)
* Sentiment Analysis (Positive / Neutral / Negative)
* Suspicious Word Detection
* AI Explanation (Why the review is fake)
* Product Trust Score
* Data Visualization (Pie Chart & Graphs)
* Review History Tracking

---

## 🧠 Technologies Used

* Python
* Streamlit
* Scikit-learn
* TF-IDF Vectorization
* TextBlob
* Matplotlib

---

## 📂 Project Structure

FakeReviewDetector/

* app.py
* backend.py
* model.pkl
* vectorizer.pkl
* requirements.txt

---

## ⚙️ Installation & Setup

1. Clone the repository
   git clone (https://github.com/SHUBU-07/fake-review-detector)
   cd fake-review-detector

2. Create virtual environment
   python -m venv .venv

3. Activate environment
   ..venv\Scripts\activate

4. Install dependencies
   pip install -r requirements.txt

5. Download TextBlob data
   python -m textblob.download_corpora

6. Run the app
   streamlit run app.py

---

## 🌐 Deployment

This project can be deployed using Streamlit Cloud.

---

## 🎯 How It Works

1. User enters a review
2. Text is converted using TF-IDF
3. Machine Learning model predicts fake/genuine
4. Sentiment analysis is performed
5. System provides explanation and visualization

---

## 📌 Future Improvements

* URL-based review analysis
* Bulk review analysis (CSV upload)
* User login system
* Advanced AI models (BERT / LSTM)

---

## 👨‍💻 Author

Shubham Baliarsingh

---

## ⭐ If you like this project

Give it a star on GitHub!
