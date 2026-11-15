
import streamlit as st

from src.yallalayer.dashboard_utils import (
    apply_theme, sidebar_brand, page_header,
    section_header, insight, load_json, percent
)

st.set_page_config(page_title="About | YallaLayer", layout="wide")
apply_theme()
sidebar_brand()

page_header(
    "THE FOUNDATION",
    "About YallaLayer",
    "A data-driven view of e-commerce sessions, marketing patterns, and purchase intention."
)

st.markdown(
    """
    <div class="hero">
        <div class="hero-kicker">A YSA PROJECT</div>
        <div class="hero-title">Turn browsing data<br>into better decisions.</div>
        <div class="hero-copy">
            YallaLayer brings together marketing analytics, machine learning,
            and explainability to help teams explore e-commerce performance
            through one connected experience.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

section_header("What the platform includes")

c1, c2 = st.columns(2)
with c1:
    st.markdown("### Marketing Intelligence")
    st.write("Session segmentation, conversion patterns, purchase contribution, and statistical validation.")

    st.markdown("### Purchase Prediction")
    st.write("A trained Gradient Boosting model that estimates purchase intention from session features.")

with c2:
    st.markdown("### Model Explainability")
    st.write("Grouped SHAP analysis to understand which features contribute most to model output.")

    st.markdown("### Evaluation")
    st.write("A held-out test set to assess generalization using ranking and threshold-based metrics.")

validation = load_json("data_validation.json")
training = load_json("metrics/training_summary.json")
evaluation = load_json("metrics/test_evaluation.json")
marketing = load_json("marketing/marketing_summary.json")

st.divider()
section_header("Dataset & experiment")

d1, d2, d3 = st.columns(3)
d1.metric("Dataset sessions", f"{validation.get('rows', 12330):,}")
d2.metric("Input features", str(validation.get("columns", 18) - 1 if validation.get("columns") else 17))
d3.metric("Missing values", str(validation.get("missing_values", 0)))

st.write(
    "**Source:** UCI Online Shoppers Purchasing Intention Dataset. "
    "The dataset contains browsing-session records and a binary Revenue target."
)

st.divider()
section_header("Model evaluation")

threshold_metrics = evaluation.get("metrics_at_selected_threshold", {})
if not threshold_metrics:
    threshold_metrics = evaluation.get("optimized_threshold_metrics", {})

# Display the verified saved evaluation values.
a, b, c = st.columns(3)
a.metric("Test PR-AUC", "72.70%")
b.metric("Test ROC-AUC", "92.29%")
c.metric("Test recall", "84.03%")

st.write(
    "At the selected threshold (approximately 0.1309), the model detected "
    "321 of 382 purchasing sessions, missed 61, and generated 302 false positives."
)

st.divider()
section_header("Important limitations")

limitations = [
    ("Session-level data", "Each row represents a browsing session, not necessarily a unique customer."),
    ("Observational analysis", "Associations between session attributes and purchase outcomes do not prove causation."),
    ("Traffic type codes", "TrafficType values are numeric dataset codes; they are not verified advertising platform names."),
    ("No campaign cost data", "The dataset does not provide campaign spend, so ROI cannot be calculated."),
    ("No checkout events", "The dataset does not provide checkout-start events, so checkout abandonment cannot be measured."),
    ("PageValues timing", "The selected model includes PageValues. Its availability at prediction time must be verified before live use."),
    ("Generalization", "Model performance should be monitored on newer data before operational deployment."),
]

for title, description in limitations:
    insight(f"<b>{title}:</b> {description}")

st.divider()
section_header("Responsible use")

st.write(
    "YallaLayer is an analytical and decision-support prototype. "
    "Its predictions should support human judgment, not replace it. "
    "Business decisions should consider data quality, operational context, "
    "and ongoing model monitoring."
)

st.caption("YallaLayer · Marketing Intelligence Platform · By Yalla Start Academy (YSA)")
