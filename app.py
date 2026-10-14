
# ---------------- PREDICTION ----------------
import html

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
                result_description = "Classified as Fake"
            else:
                result_title = "✅ Real News"
                result_description = "Classified as Real"

            # Calculate keyword influences
            keyword_items = []

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
                        key=lambda i: abs(contributions[i]),
                        reverse=True,
                    )[:5]

                    for i in ranked_indices:
                        word = html.escape(str(feature_names[i]))
                        impact = contributions[i]

                        if impact < 0:
                            label = "Associated with Fake"
                            symbol = "🔴"
                        elif impact > 0:
                            label = "Associated with Real"
                            symbol = "🟢"
                        else:
                            label = "Neutral"
                            symbol = "⚪"

                        keyword_items.append(
                            f"""
                            <div class="tc-keyword">
                                <strong>{symbol} {word}</strong>
                                <span>{label}</span>
                            </div>
                            """
                        )

            # Dashboard heading
            st.markdown("## Analysis Results")

            # Three-column dashboard
            result_col, confidence_col, keyword_col = st.columns(
                [1, 1, 1.25],
                gap="medium",
            )

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
                            Confidence in the predicted class
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.progress(
                    max(0.0, min(1.0, confidence / 100.0))
                )

            with keyword_col:
                keyword_html = "".join(keyword_items)

                if not keyword_html:
                    keyword_html = (
                        '<p class="tc-small">'
                        'No keyword influences available.</p>'
                    )

                st.markdown(
                    f"""
                    <div class="tc-card">
                        <div class="tc-label">
                            KEYWORD INFLUENCE
                        </div>
                        {keyword_html}
                        <p class="tc-small">
                            Learned word associations, not proof.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.markdown(
                """
                <style>
                .tc-card {
                    background: #111f33;
                    border: 1px solid #24415e;
                    border-radius: 14px;
                    padding: 18px;
                    min-height: 190px;
                    overflow-wrap: anywhere;
                }

                .tc-label {
                    color: #64d9ff;
                    font-size: 13px;
                    font-weight: 700;
                    margin-bottom: 14px;
                }

                .tc-value {
                    color: #ffffff;
                    font-size: 23px;
                    font-weight: 700;
                    margin: 10px 0;
                }

                .tc-small {
                    color: #b7c7dc;
                    font-size: 12px;
                    margin-top: 12px;
                }

                .tc-keyword {
                    border-bottom: 1px solid #29415a;
                    padding: 9px 0;
                    font-size: 13px;
                }

                .tc-keyword strong {
                    display: block;
                    color: #f0f6ff;
                    margin-bottom: 4px;
                }

                .tc-keyword span {
                    color: #b7c7dc;
                    font-size: 11px;
                }
                </style>
                """,
                unsafe_allow_html=True,
            )

            st.info(
                "Important: This model classifies headlines based on "
                "patterns learned during training. It does not independently "
                "verify whether a story is true. Check reliable sources."
            )

        except Exception as error:
            st.error(f"Analysis failed: {error}")


