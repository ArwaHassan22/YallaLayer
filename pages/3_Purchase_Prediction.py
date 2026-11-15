
import joblib
import pandas as pd
import streamlit as st

from src.yallalayer.dashboard_utils import (
    MODEL_PATH, apply_theme, sidebar_brand, page_header,
    section_header, insight, percent
)

st.set_page_config(page_title="Purchase Prediction | YallaLayer", layout="wide")
apply_theme()
sidebar_brand()

page_header(
    "PREDICT THE NEXT MOVE",
    "Purchase Prediction",
    "Estimate the likelihood that a browsing session will result in a purchase."
)

@st.cache_resource
def load_model_bundle():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)

bundle = load_model_bundle()

if bundle is None:
    st.error("The trained model was not found at models/yallalayer_model.joblib.")
    st.stop()

model = bundle["model"]
threshold = float(bundle.get("threshold", 0.1308793166476187))
features = bundle.get("features", [])

if not features:
    st.error("The saved model bundle does not contain its expected feature list.")
    st.stop()

section_header("Session profile", "Enter the browsing-session attributes, then run the prediction.")

with st.form("prediction_form"):
    st.markdown("**Browsing activity**")
    c1, c2, c3 = st.columns(3)

    with c1:
        administrative = st.number_input("Administrative pages", 0, 100, 0)
        informational = st.number_input("Informational pages", 0, 100, 0)
        product_related = st.number_input("Product-related pages", 0, 500, 5)

    with c2:
        administrative_duration = st.number_input("Administrative duration", 0.0, 100000.0, 0.0)
        informational_duration = st.number_input("Informational duration", 0.0, 100000.0, 0.0)
        product_duration = st.number_input("Product-related duration", 0.0, 200000.0, 120.0)

    with c3:
        bounce_rates = st.number_input("Bounce rate", 0.0, 1.0, 0.02, step=0.01, format="%.4f")
        exit_rates = st.number_input("Exit rate", 0.0, 1.0, 0.04, step=0.01, format="%.4f")
        page_values = st.number_input("PageValues", 0.0, 5000.0, 0.0)

    st.markdown("**Session context**")
    d1, d2, d3 = st.columns(3)

    with d1:
        month = st.selectbox(
            "Month",
            ["Feb", "Mar", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
            index=8,
        )
        special_day = st.number_input("SpecialDay", 0.0, 1.0, 0.0, step=0.1)

    with d2:
        operating_system = st.selectbox("Operating system code", list(range(1, 9)), index=1)
        browser = st.selectbox("Browser code", list(range(1, 14)), index=1)
        region = st.selectbox("Region code", list(range(1, 10)), index=0)

    with d3:
        traffic_type = st.selectbox("Traffic type code", list(range(1, 21)), index=1)
        visitor_type = st.selectbox(
            "Visitor type",
            ["Returning_Visitor", "New_Visitor", "Other"],
        )
        weekend = st.selectbox("Weekend session", [False, True], format_func=lambda x: "Yes" if x else "No")

    submitted = st.form_submit_button("Predict purchase intention", use_container_width=True)

if submitted:
    row = {
        "Administrative": administrative,
        "Administrative_Duration": administrative_duration,
        "Informational": informational,
        "Informational_Duration": informational_duration,
        "ProductRelated": product_related,
        "ProductRelated_Duration": product_duration,
        "BounceRates": bounce_rates,
        "ExitRates": exit_rates,
        "PageValues": page_values,
        "SpecialDay": special_day,
        "Month": month,
        "OperatingSystems": operating_system,
        "Browser": browser,
        "Region": region,
        "TrafficType": traffic_type,
        "VisitorType": visitor_type,
        "Weekend": weekend,
    }

    try:
        input_df = pd.DataFrame([row]).reindex(columns=features)
        probability = float(model.predict_proba(input_df)[0][1])
        prediction = probability >= threshold

        st.divider()
        section_header("Prediction result", "The result is based on the selected model and its saved decision threshold.")

        result_col1, result_col2 = st.columns([1, 1])

        with result_col1:
            st.metric("Purchase probability", percent(probability))
            st.progress(min(max(probability, 0.0), 1.0))

        with result_col2:
            if prediction:
                st.success("Predicted outcome: Purchase")
                st.write("The predicted probability is at or above the selected threshold.")
            else:
                st.info("Predicted outcome: No Purchase")
                st.write("The predicted probability is below the selected threshold.")

        st.caption(f"Decision threshold: {threshold:.4f}")
        insight(
            "<b>How to read this:</b> The probability is a model estimate, not a guarantee "
            "that the visitor will or will not purchase."
        )

    except Exception as exc:
        st.error(f"Prediction failed: {exc}")

st.divider()
section_header("Model operating point")
m1, m2, m3 = st.columns(3)
m1.metric("Model", bundle.get("model_name", "Gradient Boosting"))
m2.metric("Decision threshold", f"{threshold:.4f}")
m3.metric("PageValues included", "Yes" if bundle.get("page_values_included", False) else "No")

insight(
    "<b>Operational limitation:</b> The selected trained model includes PageValues. "
    "Before using this predictor in a live pre-purchase workflow, verify that PageValues "
    "is available at the exact prediction time and does not introduce target leakage."
)
