
            # ------------------------------------------
            # BALANCED RESULT CARDS - FIXED
            # ------------------------------------------
            result_col, confidence_col = st.columns(
                2,
                gap="medium"
            )

            if prediction == 0:
                result_title = "🔴 Fake News"
                result_color = "#FF7777"
            else:
                result_title = "🟢 Real News"
                result_color = "#35D5A0"

            with result_col:
                st.markdown(
                    f"""
                    <div class="tc-card">
                        <div class="tc-label">Model Prediction</div>
                        <div style="color:{result_color};
                                    font-size:1.55rem;
                                    font-weight:750;
                                    margin-top:12px;">
                            {result_title}
                        </div>
                        <div class="tc-muted" style="margin-top:8px;">
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
                        <div class="tc-label">Model Confidence</div>
                        <div style="color:#35D5F2;
                                    font-size:1.8rem;
                                    font-weight:750;
                                    margin-top:10px;">
                            {confidence:.2f}%
                        </div>
                        <div class="tc-muted" style="margin-top:8px;">
                            Estimated class probability
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.progress(
                    min(100, max(0, int(round(confidence))))
                )



