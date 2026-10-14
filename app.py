


import streamlit as st
import joblib
import pandas as pd
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

        prediction = int(model.predict([text])[0])
        probabilities = model.predict_proba([text])[0]

        classes = list(model.classes_)
        predicted_index = classes.index(prediction)
        confidence = float(probabilities[predicted_index]) * 100

        if prediction == 0:
            st.error("🔴 Prediction: Fake News")
        else:
            st.success("🟢 Prediction: Real News")

        st.metric(
            "Model Confidence",
            f"{confidence:.2f}%"
        )
        st.progress(int(round(confidence)))

        st.caption(
            "This percentage is the model's estimated probability "
            "for its selected class, not the probability that the "
            "news is factually true."
        )

        # Explain influential words using TF-IDF
        # values and Logistic Regression coefficients.
        try:
            tfidf = model.named_steps["tfidf"]
            classifier = model.named_steps["classifier"]

            vector = tfidf.transform([text])
            feature_names = tfidf.get_feature_names_out()
            coefficients = classifier.coef_[0]

            contributions = vector.multiply(coefficients).toarray()[0]

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
                    "Keyword / Feature": feature_names[idx],
                    "Influence": abs(contribution),
                    "Direction": direction
                })

            rows.sort(
                key=lambda item: item["Influence"],
                reverse=True
            )

            st.subheader("🔍 Keyword Influence")

            if rows:
                top_rows = rows[:5]

                for item in top_rows:
                    st.write(
                        f"**{item['Keyword / Feature']}** "
                        f"→ {item['Direction']}"
                    )
            else:
                st.write(
                    "No strong matching keyword features were found."
                )

            st.caption(
                "Keyword influence shows how words contribute to "
                "the model's learned classification. It does not "
                "prove a news story is fake or real."
            )

        except Exception as error:
            st.warning(
                "Keyword explanation is unavailable for this model."
            )

        st.subheader("⚠️ Verification Reminder")
        st.write(
            "Check important claims against reliable news sources, "
            "official statements, and independent fact-checkers."
        )
