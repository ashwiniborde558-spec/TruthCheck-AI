


import streamlit as st
import joblib
from pathlib import Path

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="TruthCheck | AI News Analysis",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="collapsed"
)

MODEL_PATH = Path(__file__).parent / "truthcheck_model.pkl"


# --------------------------------------------------
# LOAD EXISTING MODEL - NO RETRAINING
# --------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


model = load_model()


# --------------------------------------------------
# DARK NAVY + CYAN UI
# --------------------------------------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #07111F 0%, #0B1728 55%, #101E32 100%);
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
    background: rgba(15, 33, 53, 0.92);
    border: 1px solid #263E58;
    border-radius: 16px;
    padding: 22px;
    margin: 10px 0;
    min-height: 125px;
}

.tc-label {
    color: #91A9C4;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 1.3px;
    text-transform: uppercase;
}

.tc-value {
    color: #35D5F2;
    font-size: 1.7rem;
    font-weight: 750;
    margin-top: 12px;
}

.tc-muted {
    color: #9FB2C9;
    font-size: 0.88rem;
    line-height: 1.6;
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

div[data-testid="stMetric"] {
    background: #0F2135;
    border: 1px solid #263E58;
    padding: 18px;
    border-radius: 14px;
}

div[data-testid="stMetricLabel"] {
    color: #9FB2C9;
}

div[data-testid="stMetricValue"] {
    color: #35D5F2;
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


# --------------------------------------------------
# HEADER
# --------------------------------------------------
header_left, header_right = st.columns([5, 1])

with header_left:
    st.markdown(
        '<div class="tc-eyebrow">AI-powered news analysis</div>',
        unsafe_allow_html=True
    )
    st.markdown("# 🔎 TruthCheck")
    st.markdown(
        '<div class="tc-subtitle">'
        'Analyze a news headline with Machine Learning. '
        'Explore the prediction and confidence score.'
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


# --------------------------------------------------
# HEADLINE INPUT
# --------------------------------------------------
st.markdown("## Analyze a headline")
st.markdown(
    '<p class="tc-muted">Paste the headline you want to analyze below.</p>',
    unsafe_allow_html=True
)

headline = st.text_area(
    "News headline",
    placeholder="Example: Scientists announce a new research discovery...",
    height=125,
    label_visibility="collapsed"
)

analyze = st.button("🔍  Analyze News", type="primary")


# --------------------------------------------------
# PREDICTION
# --------------------------------------------------
if analyze:

    if not headline.strip():
        st.warning("Please enter a headline first.")

    else:
        text = headline.strip()

        try:
            prediction = int(model.predict([text])[0])
            probabilities = model.predict_proba([text])[0]

            classes = list(model.classes_)
            predicted_index = classes.index(prediction)
            confidence = float(probabilities[predicted_index]) * 100

            # ------------------------------------------
            # ANALYSIS RESULTS: TWO COLUMNS ONLY
            # ------------------------------------------
            st.markdown("## Analysis Results")

            result_col, confidence_col = st.columns(2, gap="medium")

            with result_col:
                if prediction == 0:
                    prediction_text = "🔴 Fake News"
                    prediction_color = "#FF7777"
                else:
                    prediction_text = "🟢 Real News"
                    prediction_color = "#35D5A0"

                st.markdown(
                    f"""
                    <div class="tc-card">
                        <div class="tc-label">MODEL PREDICTION</div>
                        <div style="
                            color:{prediction_color};
                            font-size:1.7rem;
                            font-weight:750;
                            margin-top:12px;
                        ">{prediction_text}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with confidence_col:
                st.markdown(
                    f"""
                    <div class="tc-card">
                        <div class="tc-label">MODEL CONFIDENCE</div>
                        <div class="tc-value">{confidence:.2f}%</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            # Confidence progress bar
            st.progress(
                min(1.0, max(0.0, confidence / 100))
            )

            # Confidence message
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
            # VERIFICATION REMINDER
            # ------------------------------------------
            st.markdown("## ⚠️ Verification Reminder")

            st.markdown(
                '<div class="tc-card">'
                '<div class="tc-muted">'
                'Check important claims against reliable news sources, '
                'official statements, and independent fact-checkers. '
                'TruthCheck does not independently verify news facts.'
                '</div></div>',
                unsafe_allow_html=True
            )

        except Exception as error:
            st.error("Unable to analyze this headline.")
            st.exception(error)


# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.divider()

st.markdown(
    '<div style="text-align:center;color:#7187A1;font-size:0.8rem;">'
    'TruthCheck · Machine Learning Project · '
    'Predictions require independent verification'
    '</div>',
    unsafe_allow_html=True
)
