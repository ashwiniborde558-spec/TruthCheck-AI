

from pathlib import Path

import joblib
import streamlit as st

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="TruthCheck AI",
    page_icon="🔎",
    layout="wide",
)

# ---------------- CUSTOM DESIGN ----------------
st.markdown(
    """
    <style>
    .stApp {
        background-color: #08111f;
        color: #e8f1ff;
    }

    .block-container {
        max-width: 1100px;
        padding-top: 2rem;
    }

    h1, h2, h3, p, label {
        color: #e8f1ff;
    }

    .tc-card {
        background: #111f33;
        border: 1px solid #24415e;
        border-radius: 16px;
        padding: 24px;
        min-height: 145px;
        margin-bottom: 16px;
        overflow-wrap: anywhere;
    }

    .tc-label {
        color: #64d9ff;
        font-size: 14px;
        font-weight: 600;
        margin-bottom: 12px;
    }

    .tc-value {
        color: #ffffff;
        font-size: 27px;
        font-weight: 700;
        margin: 0;
    }

    .tc-small {
        color: #b7c7dc;
        font-size: 14px;
        margin-top: 10px;
    }

    div.stButton > button {
        background-color: #00b8d9;
        color: #06111f;
        border: none;
        border-radius: 10px;
        font-weight: 700;
        min-height: 45px;
        width: 100%;
    }

    div.stButton > button:hover {
        background-color: #65e2ff;
        color: #06111f;
    }

    textarea {
        background-color: #111f33 !important;
        color: #ffffff !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------- LOAD MODEL ----------------
MODEL_PATH = Path(__file__).parent / "truthcheck_model.pkl"


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH.name}. "
            "Ensure it is in the same repository folder as app.py."
        )
    return joblib.load(MODEL_PATH)


try:
    model = load_model()
except Exception as error:
    st.error(f"Could not load the model: {error}")
    st.stop()

# ---------------- HEADER ----------------
st.markdown(
    """
    <h1 style="text-align:center;">🔎 TruthCheck AI</h1>
    <p style="text-align:center; color:#64d9ff;">
        AI-powered news headline classification
    </p>
    """,
    unsafe_allow_html=True,
)

st.write(
    "Enter a news headline below to classify it using the trained machine-learning model."
)

# ---------------- INPUT ----------------
headline = st.text_area(
    "News headline",
    placeholder="Paste a news headline here...",
    height=150,
    key="headline_input",
)

analyze = st.button("Analyze News", use_container_width=True)

# ---------------- PREDICTION ----------------
if analyze:
    text = headline.strip()

    if not text:
        st.warning("Please enter a headline before analyzing.")
    else:
        try:
            prediction = int(model.predict([text])[0])

            probabilities = model.predict_proba([text])[0]
            classes = list(model.classes_)
            predicted_index = classes.index(prediction)

            confidence = float(probabilities[predicted_index]) * 100

            if prediction == 0:
                result_title = "⚠️ Fake News"
                result_description = (
                    "The model classified this headline as fake."
                )
            else:
                result_title = "✅ Real News"
                result_description = (
                    "The model classified this headline as real."
                )

            st.markdown("## Analysis Results")

            result_col, confidence_col = st.columns(2, gap="medium")

            with result_col:
                st.markdown(
                    f"""
                    <div class="tc-card">
                        <div class="tc-label">PREDICTION</div>
                        <p class="tc-value">{result_title}</p>
                        <p class="tc-small">{result_description}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with confidence_col:
                st.markdown(
                    f"""
                    <div class="tc-card">
                        <div class="tc-label">MODEL CONFIDENCE</div>
                        <p class="tc-value">{confidence:.2f}%</p>
                        <p class="tc-small">
                            Confidence in the predicted class, not proof of truth.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.progress(
                max(0.0, min(1.0, confidence / 100.0))
            )

            # -------- WORD INFLUENCE --------
            if (
                hasattr(model, "named_steps")
                and "tfidf" in model.named_steps
                and "classifier" in model.named_steps
            ):
                tfidf = model.named_steps["tfidf"]
                classifier = model.named_steps["classifier"]

                if hasattr(classifier, "coef_"):
                    vector = tfidf.transform([text])
                    feature_names = tfidf.get_feature_names_out()
                    coefficients = classifier.coef_[0]

                    contributions = (
                        vector.multiply(coefficients).toarray()[0]
                    )

                    present_indices = vector.nonzero()[1]

                    ranked_indices = sorted(
                        present_indices,
                        key=lambda index: abs(contributions[index]),
                        reverse=True,
                    )[:5]

                    st.markdown("### Words influencing the prediction")

                    if ranked_indices:
                        for index in ranked_indices:
                            word = feature_names[index]
                            impact = contributions[index]

                            if impact > 0:
                                direction = "associated with Real"
                            elif impact < 0:
                                direction = "associated with Fake"
                            else:
                                direction = "neutral"

                            st.write(
                                f"- **{word}** — {direction}"
                            )
                    else:
                        st.info(
                            "No recognized headline features were found."
                        )

            # -------- DISCLAIMER --------
            st.info(
                "Important: This is a machine-learning classification, "
                "not independent fact-checking. The model was trained on "
                "news headlines and may make mistakes. Verify claims using "
                "reliable news sources and original evidence."
            )

        except Exception as error:
            st.error(f"Analysis failed: {error}")
