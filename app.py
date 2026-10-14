import streamlit as st
import joblib
from pathlib import Path

st.set_page_config(
    page_title="TruthCheck",
    page_icon="🔎",
    layout="centered"
)

MODEL_PATH = Path(__file__).parent / "truthcheck_model.pkl"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

st.title("🔎 TruthCheck")
st.subheader("AI-Powered Fake News Detection")

st.write(
    "Enter a news headline to classify it as Fake or Real. "
    "This is a prediction, not a verified fact-check."
)

headline = st.text_area(
    "Enter News Headline",
    placeholder="Type or paste a news headline here..."
)

if st.button("Check News", type="primary"):
    if not headline.strip():
        st.warning("Please enter a headline first.")
    else:
        prediction = int(model.predict([headline.strip()])[0])

        if prediction == 0:
            st.error("Prediction: Fake News")
        else:
            st.success("Prediction: Real News")

        st.caption(
            "This model analyzes headlines and may make mistakes. "
            "Verify important claims using reliable sources."
        )
