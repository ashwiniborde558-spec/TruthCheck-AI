
import streamlit as st
import joblib
from pathlib import Path

# ==================================================
# 1. PAGE CONFIGURATION
# ==================================================
st.set_page_config(
    page_title="TruthCheck | AI News Analysis",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="collapsed"
)

MODEL_PATH = Path(__file__).parent / "truthcheck_model.pkl"


# ==================================================
# 2. LOAD THE EXISTING ML MODEL
# ==================================================
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


model = load_model()


# ==================================================
# 3. DARK NAVY AND CYAN UI
# ==================================================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(
        135deg, #07111F 0%, #0B1728 55%, #101E32 100%
    );
    color: #EAF2FF;
}

.block-container {
    max-width: 1050px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

[data-testid="stHeader"] {
    background: transparent;
}

h1, h2, h3 {
    color: #F2F7FF !important;
}

p, label, .stMarkdown {
    color: #C3D2E5;
}

.tc-eyebrow {
    color: #35D5F2;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
}

.tc-subtitle {
    color: #A7BAD1;
    font-size: 1rem;
    line-height: 1.7;
}

.tc-card {
    box-sizing: border-box;
    background: #0F2135;
    border: 1px solid #263E58;
    border-radius: 16px;
    padding: 20px;
    min-height: 145px;
    height: 100%;
    margin: 0;
}

.tc-label {
    color: #91A9C4;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 1.3px;
    text-transform: uppercase;
}

.tc-muted {
    color: #9FB2C9;
    font-size: 0.88rem;
    line-height: 1.6;
}

.tc-keyword {
    background: #122941;
    border: 1px solid #284763;
    border-radius: 10px;
    padding: 12px 14px;
    margin: 7px 0;
    color: #EAF2FF;
    overflow-wrap: anywhere;
}

div.stButton > button[kind="primary"] {
    background: #35D5F2;
    color: #06111F;
    border: none;
    border-radius: 10px;
    min-height: 3rem;
    font-weight: 750;
    width: 100%;
    transition: 0.2s ease;
}

div.stButton > button[kind="primary"]:hover {
    background: #80E9FA;
    color: #06111F;
    border: none;
}

.stTextArea textarea {
    background: #0B192A !important;
    color: #F2F7FF !important;
    border: 1px solid #304861 !important;
    border-radius: 12px !important;
}

.stTextArea textarea:focus {
    border-color: #35D5F2 !important;
    box-shadow: 0 0 0 1px #35D5F2 !important;
}

div[data-testid="stProgressBar"] > div > div {
    background-color: #35D5F2;
}

hr {
    border-color: #263E58;
}

footer {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)


# ==================================================
# 4. HEADER
# ==================================================
header_left, header_right = st.columns([5, 1])

with header_left:
    st.markdown(
        '<div class="tc-eyebrow">'
        'AI-powered news analysis'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown("# 🔎 TruthCheck")

    st.markdown(
        '<div class="tc-subtitle">'
        'Analyze a news headline with Machine Learning. '
        'Explore the prediction, confidence score, and '
        'influential words.'
        '</div>',
        unsafe_allow_html=True
    )

with header_right:
    st.markdown(
        '<div class="tc-card" style="text-align:center;">'
        '<div style="font-size:2rem;">🛡️</div>'
        '<div style="color:#35D5F2;font-weight:700;">ML</div>'
        '<div class="tc-muted">Analysis</div>'
        '</div>',
        unsafe_allow_html=True
    )

st.divider()


# ==================================================
# 5. HEADLINE INPUT
# ==================================================
st.markdown("## Analyze a headline")

st.markdown(
    '<p class="tc-muted">'
    'Paste the headline you want to analyze below.'
    '</p>',
    unsafe_allow_html=True
)

headline = st.text_area(
    "News headline",
    placeholder="Enter a news headline here...",
    height=125,
    label_visibility="collapsed"
)

analyze = st.button(
    "🔍  Analyze News",
    type="primary"
)


# ==================================================
# 6. PREDICTION AND RESULTS
# ==================================================
if analyze:

    if not headline.strip():
        st.warning("Please enter a headline first.")

    else:
        text = headline.strip()

        try:
            # Keep the existing model prediction logic
            prediction = int(model.predict([text])[0])
            probabilities = model.predict_proba([text])[0]

            classes = list(model.classes_)
            predicted_index = classes.index(prediction)

            confidence = (
                float(probabilities[predicted_index]) * 100
            )

            # ------------------------------------------
            # RESULT CARD CONTENT
            # ------------------------------------------
            if prediction == 0:
                result_title = "🔴 Fake News"
                result_color = "#FF7777"
            else:
                result_title = "🟢 Real News"
                result_color = "#35D5A0"

            st.markdown("## Analysis Results")

            # ------------------------------------------
            # ALIGNED PREDICTION AND CONFIDENCE CARDS
            # ------------------------------------------
            result_col, confidence_col = st.columns(
                2,
                gap="medium"
            )

            with result_col:
                st.markdown(
                    f"""
                    <div class="tc-card">
                        <div class="tc-label">
                            Model Prediction
                        </div>

                        <div style="
                            color: {result_color};
                            font-size: 1.55rem;
                            font-weight: 750;
                            margin-top: 12px;
                        ">
                            {result_title}
                        </div>

                        <div class="tc-muted" style="margin-top: 8px;">
                            Model classification
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with confidence_col:
                st.markdown(
                    f"""
                    <div class="tc-card">
                        <div class="tc-label">
                            Model Confidence
                        </div>

                        <div style="
                            color: #35D5F2;
                            font-size: 1.8rem;
                            font-weight: 750;
                            margin-top: 10px;
                        ">
                            {confidence:.2f}%
                        </div>

                        <div class="tc-muted" style="margin-top: 8px;">
                            Estimated class probability
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            # Progress bar outside the cards
            st.progress(
                min(100, max(0, int(round(confidence))))
            )

            # ------------------------------------------
            # CONFIDENCE MESSAGE
            # ------------------------------------------
            if confidence < 60:
                st.warning(
                    "Low confidence: The model is uncertain "
                    "about this prediction."
                )

            elif confidence < 80:
                st.info(
                    "Moderate confidence: Verify this headline "
                    "using reliable sources."
                )

            else:
                st.success(
                    "Higher model confidence, but independent "
                    "verification is still needed."
                )

            st.caption(
                "This score reflects the model's prediction, "
                "not whether the news is factually true."
            )

            # ------------------------------------------
            # KEYWORD INFLUENCE
            # ------------------------------------------
            st.markdown("## 🔍 Keyword Influence")

            st.markdown(
                '<p class="tc-muted">'
                'Words associated with the model’s learned classification.'
                '</p>',
                unsafe_allow_html=True
            )

            try:
                tfidf = model.named_steps["tfidf"]
                classifier = model.named_steps["classifier"]

                vector = tfidf.transform([text])
                feature_names = tfidf.get_feature_names_out()
                coefficients = classifier.coef_[0]

                contributions = (
                    vector.multiply(coefficients).toarray()[0]
                )

                present_indices = vector.nonzero()[1]

                rows = []

                for idx in present_indices:
                    contribution = float(contributions[idx])

                    if contribution > 0:
                        direction = "Real News"

                    elif contribution < 0:
                        direction = "Fake News"

                    else:
                        continue

                    rows.append({
                        "keyword": feature_names[idx],
                        "influence": abs(contribution),
                        "direction": direction
                    })

                rows.sort(
                    key=lambda item: item["influence"],
                    reverse=True
                )

                if rows:
                    for item in rows[:5]:

                        if item["direction"] == "Real News":
                            color = "#35D5A0"
                        else:
                            color = "#FF8585"

                        st.markdown(
                            f'<div class="tc-keyword">'
                            f'<span style="font-weight:700;">'
                            f'{item["keyword"]}</span>'
                            f'<span style="float:right;color:{color};'
                            f'font-weight:700;">'
                            f'{item["direction"]}</span>'
                            f'</div>',
                            unsafe_allow_html=True
                        )

                else:
                    st.info(
                        "No matching keyword features were found."
                    )

                st.caption(
                    "These are model-specific word associations, "
                    "not proof that a story is fake or real."
                )

            except Exception:
                st.warning(
                    "Keyword explanation is unavailable for this model."
                )

            # ------------------------------------------
            # VERIFICATION REMINDER
            # ------------------------------------------
            st.markdown("## ⚠️ Verification Reminder")

            st.markdown(
                '<div class="tc-card" style="min-height:0;">'
                '<div class="tc-muted">'
                'Check important claims against reliable news sources, '
                'official statements, and independent fact-checkers. '
                'TruthCheck does not independently verify news facts.'
                '</div>'
                '</div>',
                unsafe_allow_html=True
            )

        except Exception as error:
            st.error("Unable to analyze this headline.")
            st.exception(error)


# ==================================================
# 7. FOOTER
# ==================================================
st.divider()

st.markdown(
    '<div style="text-align:center;color:#7187A1;'
    'font-size:0.8rem;">'
    'TruthCheck · Machine Learning Project · '
    'Predictions require independent verification'
    '</div>',
    unsafe_allow_html=True
)

           

