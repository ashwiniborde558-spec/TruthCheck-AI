

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
    "Analyze a news headline using Machine Learning. "
    "View the prediction, confidence score, and influential keywords."
)

st.info(
    "This tool predicts patterns learned from its training dataset. "
    "It does not independently verify whether a claim is true."
)

headline = st.text_area(
    "Enter News Headline",
    placeholder="Type or paste a news headline here...",
    height=120
)

if st.button("Analyze News", type="primary"):
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

            if prediction == 0:
                st.error("🔴 Prediction: Fake News")
            else:
                st.success("🟢 Prediction: Real News")

            st.metric("Model Confidence", f"{confidence:.2f}%")
            st.progress(
                min(100, max(0, int(round(confidence))))
            )

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

            # Keyword influence
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

                st.subheader("🔍 Keyword Influence")

                if rows:
                    for item in rows[:5]:
                        st.write(
                            f"**{item['keyword']}** "
                            f"→ {item['direction']}"
                        )
                else:
                    st.write(
                        "No matching keyword features were found."
                    )

                st.caption(
                    "Keyword influence reflects the model's learned "
                    "word associations. It does not prove a story "
                    "is fake or real."
                )

            except Exception:
                st.warning(
                    "Keyword explanation is unavailable for this model."
                )

            st.subheader("⚠️ Verification Reminder")
            st.write(
                "Check important claims against reliable news sources, "
                "official statements, and independent fact-checkers."
            )

        except Exception as error:
            st.error("Unable to analyze this headline.")
            st.exception(error)

