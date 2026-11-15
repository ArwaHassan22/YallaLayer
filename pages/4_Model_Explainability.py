
import streamlit as st
import plotly.express as px

from src.yallalayer.dashboard_utils import (
    METRICS, apply_theme, sidebar_brand, page_header,
    section_header, insight, load_csv, load_json, plot_layout, percent
)

st.set_page_config(page_title="Model Explainability | YallaLayer", layout="wide")
apply_theme()
sidebar_brand()

page_header(
    "BEHIND THE PREDICTION",
    "Model Explainability",
    "Understand which session features contribute most to the model's predictions."
)

shap_data = load_csv(METRICS / "shap_feature_importance.csv")
shap_summary = load_json(METRICS / "shap_summary.json")
evaluation = load_json(METRICS / "test_evaluation.json")

if shap_data.empty:
    st.error("SHAP feature importance report was not found.")
    st.stop()

required = {"Feature", "Mean_Absolute_SHAP"}
if not required.issubset(shap_data.columns):
    st.error("The SHAP report does not contain the expected columns.")
    st.stop()

section_header(
    "Global feature importance",
    "Mean absolute SHAP values aggregated at the original feature level."
)

top_n = st.slider("Number of features to display", 5, min(17, len(shap_data)), min(10, len(shap_data)))

top_features = shap_data.sort_values("Mean_Absolute_SHAP", ascending=False).head(top_n)
fig = px.bar(
    top_features.sort_values("Mean_Absolute_SHAP"),
    x="Mean_Absolute_SHAP",
    y="Feature",
    orientation="h",
    color="Mean_Absolute_SHAP",
    color_continuous_scale=["#DCE8F8", "#1769D2", "#101D36"],
    title="Global feature importance",
    labels={"Mean_Absolute_SHAP": "Mean |SHAP value|"},
)
st.plotly_chart(plot_layout(fig, max(390, top_n * 38)), use_container_width=True)

if "Mean_Signed_SHAP" in shap_data.columns:
    section_header("Contribution direction", "Average signed grouped SHAP values. Positive and negative values can offset each other.")
    direction = shap_data.sort_values("Mean_Signed_SHAP")
    fig2 = px.bar(
        direction,
        x="Mean_Signed_SHAP",
        y="Feature",
        orientation="h",
        color="Mean_Signed_SHAP",
        color_continuous_scale=["#D97757", "#F2F4F7", "#1769D2"],
        title="Average signed SHAP contribution",
        labels={"Mean_Signed_SHAP": "Mean signed SHAP"},
    )
    st.plotly_chart(plot_layout(fig2, max(390, len(direction) * 30)), use_container_width=True)

section_header("Top drivers", "Features with the largest average absolute contribution magnitude.")
for _, row in shap_data.sort_values("Mean_Absolute_SHAP", ascending=False).head(5).iterrows():
    st.markdown(
        f"**{row['Feature']}** — {row['Mean_Absolute_SHAP']:.4f}"
    )

st.divider()
section_header("Individual explanation", "A sample-level explanation saved from the training-set SHAP analysis.")

individual = shap_summary.get("individual_explanation", {})
if not individual:
    individual = shap_summary.get("individual_sample", {})

if individual:
    a, b, c = st.columns(3)
    probability = individual.get("purchase_probability")
    predicted = individual.get("predicted_class")
    actual = individual.get("actual_class")
    threshold = individual.get("threshold")

    a.metric("Predicted purchase probability", percent(probability) if probability is not None else "N/A")
    b.metric("Predicted class", str(predicted) if predicted is not None else "N/A")
    c.metric("Actual class", str(actual) if actual is not None else "N/A")

    if threshold is not None:
        st.caption(f"Saved decision threshold: {float(threshold):.4f}")
else:
    st.info("Individual explanation details are not available in the saved summary.")

st.divider()
section_header("Explainability methodology")

method = shap_summary.get("importance_method")
if method:
    st.write(method)
else:
    st.write(
        "TreeExplainer was used for the Gradient Boosting model. Encoded feature "
        "contributions were grouped back into their original input features."
    )

st.caption(
    f"SHAP method: {shap_summary.get('shap_method', 'TreeExplainer')} · "
    f"Samples: {shap_summary.get('sample_size', 2000)} · "
    f"Original features: {shap_summary.get('original_feature_count', 17)} · "
    f"Encoded features: {shap_summary.get('encoded_feature_count', 74)}"
)

insight(
    "<b>Interpretation:</b> SHAP values describe how features contribute to the model's "
    "output. They do not establish causal relationships. Global importance measures "
    "magnitude, while signed averages can hide different effects across categories."
)

with st.expander("View complete SHAP table"):
    st.dataframe(shap_data, use_container_width=True, hide_index=True)
