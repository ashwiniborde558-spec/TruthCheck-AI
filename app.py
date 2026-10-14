
from pathlib import Path
import html

import joblib
import streamlit as st


# ==================================================
# PAGE CONFIGURATION
# ==================================================
st.set_page_config(
    page_title="TruthCheck AI",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ==================================================
# CUSTOM CSS
# ==================================================
st.markdown(
    """
    <style>
    .stApp {
        background-color: #08111f;
        color: #e8f1ff;
    }

    .block-container {
        max-width: 1450px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    h1, h2, h3, p, label {
        color: #e8f1ff;
    }

    .tc-header {
        text-align: center;
        padding: 10px 0 25px 0;
    }

    .tc-subtitle {
        color: #64d9ff;
        font-size: 16px;
    }

    .tc-card {
        background: #111f33;
        border: 1px solid #24415e;
        border-radius: 16px;
        padding: 20px;
        min-height: 205px;
        height: 100%;
        overflow-wrap: anywhere;
    }

    .tc-label {
        color: #64d9ff;
        font-size: 13px;
        font-weight: 700;
        letter-spacing: 1px;
        margin-bottom: 18px;
    }

    .tc-value {
        color: #ffffff;
        font-size: 25px;
        font-weight: 700;
        margin: 12px 0;
    }

    .tc-description {
        color: #b7c7dc;
        font-size: 14px;
        line-height: 1.5;
        margin-top: 12px;
    }

    .tc-keyword {
        border-bottom: 1px solid #29415a;
        padding: 10px 0;
    }

    .tc-keyword:last-child {
        border-bottom: none;
    }

    .tc-word {
        color: #f0f6ff;
        font-weight: 600;
        font-size: 14px;
    }

    .tc-association {
        color: #b7c7dc;
        font-size: 12px;
        margin-top: 5px;
    }

    div.stButton > button {
        background-color: #00b8d9;
        color: #06111f;
        border: none;
        border-radius: 10px;
        font-weight: 700;
        min-height: 46px;
        width: 100%;
    }

    div.stButton > button:hover {
        background-color: #65e2ff;
        color: #06111f;
    }

    div[data-testid="stTextArea"] textarea {
        background-color: #111f33;
        color: #ffffff;
        border: 1px solid #24415e;
        border-radius: 12px;
    }

    div[data-testid="stProgressBar"] {
        margin-top: 8px;
        margin-bottom: 15px;
    }

    .tc-note {
        color: #b7c7dc;
        font-size: 13px;
        line-height: 1.6;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==================================================
# LOAD TRAINED MODEL
# ==================================================
MODEL_PATH = Path(__file__).resolve().parent / "truthcheck_model.pkl"


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "truthcheck_model.pkl was not found. "
            "Place it in the same folder as app.py."
        )

    return joblib.load(MODEL_PATH)


try:
    model = load_model()
except Exception as error:
    st.error(f"Unable to load the trained model: {error}")
    st.stop()


# ==================================================
# HEADER
# ==================================================
st.markdown(
    """
    <div class="tc-header">
        <h1>🔎 TruthCheck AI</h1>
        <p class="tc-subtitle">
            AI-Powered News Headline Classification
        </p>
        <p class="tc-note">
            Analyze a headline using a trained machine-learning model.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("### Enter a News Headline")

# ==================================================
# INPUT AND BUTTON
# ==================================================
headline = st.text_area(
    "News headline",
    placeholder=(
        "Example: Government announces a new policy "
        "to improve public transport..."
    ),
    height=130,
    key="headline_input",
    label_visibility="collapsed",
)

analyze = st.button(
    "🔍 Analyze News",
    use_container_width=True,
)


# ==================================================
# PREDICTION
# ==================================================
if analyze:

    text = headline.strip()

    if not text:
        st.warning("Please enter a news headline first.")

    else:
        try:
            # Get model prediction
            prediction = int(model.predict([text])[0])

            # Get probability for each class
            probabilities = model.predict_proba([text])[0]
            classes = list(model.classes_)

            predicted_index = classes.index(prediction)
            confidence = float(
                probabilities[predicted_index]
            ) * 100

            # Result text
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

            # ------------------------------------------
            # KEYWORD INFLUENCE CALCULATION
            # ------------------------------------------
            keyword_items = []

            if (
                hasattr(model, "named_steps")
                and "tfidf" in model.named_steps
                and "classifier" in model.named_steps
            ):
                tfidf = model.named_steps["tfidf"]
                classifier = model.named_steps["classifier"]

                if (
                    hasattr(classifier, "coef_")
                    and len(classifier.coef_) > 0
                ):
                    vector = tfidf.transform([text])

                    feature_names = (
                        tfidf.get_feature_names_out()
                    )
                    coefficients = classifier.coef_[0]

                    contributions = (
                        vector.multiply(coefficients).toarray()[0]
                    )

                    present_indices = vector.nonzero()[1]

                    ranked_indices = sorted(
                        present_indices,
                        key=lambda i: abs(contributions[i]),
                        reverse=True,
                    )[:5]

                    for index in ranked_indices:
                        word = html.escape(
                            str(feature_names[index])
                        )
                        impact = float(contributions[index])

                        if impact < 0:
                            symbol = "🔴"
                            association = "Associated with Fake"
                        elif impact > 0:
                            symbol = "🟢"
                            association = "Associated with Real"
                        else:
                            symbol = "⚪"
                            association = "Neutral influence"

                        keyword_items.append(
                            f"""
                            <div class="tc-keyword">
                                <div class="tc-word">
                                    {symbol} {word}
                                </div>
                                <div class="tc-association">
                                    {association}
                                </div>
                            </div>
                            """
                        )

            # ------------------------------------------
            # RESULTS HEADING
            # ------------------------------------------
            st.markdown("---")
            st.markdown("## Analysis Results")

            # Three side-by-side columns
            result_col, confidence_col, keyword_col = st.columns(
                [1, 1, 1.15],
                gap="medium",
            )

            # ------------------------------------------
            # COLUMN 1: PREDICTION
            # ------------------------------------------
            with result_col:
                st.markdown(
                    f"""
                    <div class="tc-card">
                        <div class="tc-label">
                            PREDICTION
                        </div>
                        <div class="tc-value">
                            {result_title}
                        </div>
                        <div class="tc-description">
                            {result_description}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # ------------------------------------------
            # COLUMN 2: CONFIDENCE
            # ------------------------------------------
            with confidence_col:
                st.markdown(
                    f"""
                    <div class="tc-card">
                        <div class="tc-label">
                            MODEL CONFIDENCE
                        </div>
                        <div class="tc-value">
                            {confidence:.2f}%
                        </div>
                        <div class="tc-description">
                            Probability assigned to the predicted class,
                            not proof that the story is true.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.progress(
                    max(0.0, min(1.0, confidence / 100.0))
                )

            # ------------------------------------------
            # COLUMN 3: KEYWORD INFLUENCE
            # ------------------------------------------
            with keyword_col:

                if keyword_items:
                    keyword_content = "".join(keyword_items)
                else:
                    keyword_content = """
                    <div class="tc-description">
                        No recognized words were available
                        for this explanation.
                    </div>
                    """

                st.markdown(
                    f"""
                    <div class="tc-card">
                        <div class="tc-label">
                            KEYWORD INFLUENCE
                        </div>
                        {keyword_content}
                        <div class="tc-description">
                            Top words associated with the model's
                            prediction.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # ------------------------------------------
            # DISCLAIMER
            # ------------------------------------------
            st.markdown("")
            st.info(
                "Important: TruthCheck AI classifies headlines "
                "using patterns learned from its training dataset. "
                "Keyword associations are not evidence. A Real News "
                "prediction does not verify a story, and a Fake News "
                "prediction does not prove it false. Always verify "
                "important claims using reliable sources."
            )

        except Exception as error:
            st.error(
                f"Analysis failed: {type(error).__name__}: {error}"
            )
