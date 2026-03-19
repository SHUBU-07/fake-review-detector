import streamlit as st
import time
import matplotlib.pyplot as plt
from train import analyze_review

# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="Fake Review Detector", page_icon="🕵️", layout="wide")

# ------------------ SESSION STATE (History) ------------------
if "history" not in st.session_state:
    st.session_state.history = []

# ------------------ CUSTOM CSS ------------------
st.markdown("""
<style>
.main {
    background: linear-gradient(135deg, #0e1117, #1c1f26);
    color: white;
}
.stTextArea textarea {
    background-color: #262730;
    color: white;
    border-radius: 12px;
}
.stButton button {
    background: linear-gradient(90deg, #ff4b4b, #ff6b6b);
    color: white;
    border-radius: 12px;
    height: 3em;
    width: 100%;
    font-size: 18px;
}
</style>
""", unsafe_allow_html=True)

# ------------------ SIDEBAR ------------------
st.sidebar.title("📌 Dashboard")
st.sidebar.write("Fake Review Detector AI")
st.sidebar.markdown("---")
st.sidebar.write("### History")

for item in st.session_state.history[-5:][::-1]:
    st.sidebar.write(f"🔹 {item['review'][:30]}...")
    st.sidebar.write(f"➡️ {item['result']} ({item['confidence']}%)")

# ------------------ HEADER ------------------
st.markdown("<h1 style='text-align:center;'>🕵️ Fake Review Detector</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>AI-powered fake review detection system</p>", unsafe_allow_html=True)

st.divider()

# ------------------ INPUT ------------------
col1, col2 = st.columns([2,1])

with col1:
    review = st.text_area("✍️ Enter Review", height=150)
    analyze = st.button("🚀 Analyze Review")

with col2:
    st.info("💡 Tips:\n- Avoid generic phrases\n- Watch repeated words\n- Check overly positive tone")

st.divider()

# ------------------ ANALYSIS ------------------
if analyze and review.strip() != "":
    
    with st.spinner("🤖 AI analyzing..."):
        time.sleep(1.2)
        result = analyze_review(review)

    # Save history
    st.session_state.history.append({
        "review": review,
        "result": result["result"],
        "confidence": result["confidence"]
    })

    # RESULT
    if result["result"] == "Fake":
        st.error(f"⚠️ Fake Review ({result['confidence']}%)")
    else:
        st.success(f"✅ Genuine Review ({result['confidence']}%)")

    # METRICS
    col1, col2, col3 = st.columns(3)
    col1.metric("Confidence", f"{result['confidence']}%")
    col2.metric("Sentiment", result["sentiment"])
    col3.metric("Trust Score", f"{100 - result['confidence']}%")

    st.divider()

    # EXPLANATION
    st.subheader("🤖 AI Explanation")
    if result["explanation"]:
        for e in result["explanation"]:
            st.write("•", e)
    else:
        st.write("No major issues detected")

    # SUSPICIOUS WORDS
    if result["suspicious_words"]:
        st.warning("⚠️ Suspicious Words:")
        st.write(", ".join(result["suspicious_words"]))

    st.divider()

    # ------------------ CHARTS ------------------
    st.subheader("📊 Visualization")

    fake = result["confidence"]
    genuine = 100 - fake

    col1, col2 = st.columns(2)

    # PIE CHART
    with col1:
        fig1, ax1 = plt.subplots()
        ax1.pie([fake, genuine], labels=["Fake", "Genuine"], autopct='%1.1f%%')
        ax1.set_title("Fake vs Genuine")
        st.pyplot(fig1)

    # BAR CHART (Sentiment)
    with col2:
        sentiment_map = {"Positive": 1, "Neutral": 0, "Negative": -1}
        score = sentiment_map.get(result["sentiment"], 0)

        fig2, ax2 = plt.subplots()
        ax2.bar(["Sentiment Score"], [score])
        ax2.set_title("Sentiment Analysis")
        st.pyplot(fig2)

    st.divider()

# ------------------ FOOTER ------------------
st.markdown("---")
st.markdown("<p style='text-align:center;'>Made with ❤️ using Streamlit</p>", unsafe_allow_html=True)
