
import streamlit as st
import plotly.express as px

from src.yallalayer.dashboard_utils import (
    MARKETING, apply_theme, sidebar_brand, page_header,
    section_header, insight, load_csv, plot_layout, percent
)

st.set_page_config(page_title="Marketing Intelligence | YallaLayer", layout="wide")
apply_theme()
sidebar_brand()

page_header(
    "UNDERSTAND THE WHY",
    "Marketing Intelligence",
    "Explore how browsing behavior and session segments relate to purchasing decisions."
)

reports = {
    "Monthly performance": ("monthly_performance.csv", "Month"),
    "Visitor type": ("visitor_type_performance.csv", "VisitorType"),
    "Traffic source codes": ("traffic_source_performance.csv", "TrafficType"),
    "Weekday vs weekend": ("weekend_performance.csv", "Day_Type"),
    "Product-related activity": ("productrelated_level_performance.csv", "ProductRelated_Level"),
    "Product-related duration": ("productrelated_duration_level_performance.csv", "ProductRelated_Duration_Level"),
    "Bounce rate": ("bounce_rate_performance.csv", "BounceRates_Level"),
    "Exit rate": ("exit_rate_performance.csv", "ExitRates_Level"),
}

selected = st.selectbox("Choose an analysis", list(reports.keys()))
filename, category_column = reports[selected]
df = load_csv(MARKETING / filename)

if df.empty:
    st.error(f"Could not find the report: {filename}")
    st.stop()

required = {category_column, "Sessions", "Purchases", "Conversion_Rate"}
if not required.issubset(df.columns):
    st.error("The selected report does not contain the expected columns.")
    st.stop()

section_header(selected, "Compare session volume, purchase contribution, and conversion rate.")

total_sessions = df["Sessions"].sum()
total_purchases = df["Purchases"].sum()
overall_rate = total_purchases / total_sessions if total_sessions else 0

a, b, c = st.columns(3)
a.metric("Sessions in selected report", f"{total_sessions:,.0f}")
b.metric("Purchases", f"{total_purchases:,.0f}")
c.metric("Weighted conversion rate", percent(overall_rate))

left, right = st.columns([1.2, 1])

with left:
    fig = px.bar(
        df.sort_values("Conversion_Rate", ascending=False),
        x=category_column,
        y="Conversion_Rate",
        color="Conversion_Rate",
        color_continuous_scale=["#DCE8F8", "#1769D2", "#101D36"],
        hover_data=["Sessions", "Purchases"],
        title="Conversion rate by segment",
        labels={"Conversion_Rate": "Conversion rate (%)", category_column: "Segment"},
    )
    st.plotly_chart(plot_layout(fig, 390), use_container_width=True)

with right:
    if "Purchase_Contribution" in df.columns:
        fig2 = px.bar(
            df.sort_values("Purchase_Contribution", ascending=False),
            x=category_column,
            y="Purchase_Contribution",
            color_discrete_sequence=["#E9A928"],
            hover_data=["Sessions", "Purchases"],
            title="Share of all purchases",
            labels={"Purchase_Contribution": "Purchase contribution (%)"},
        )
        st.plotly_chart(plot_layout(fig2, 390), use_container_width=True)
    else:
        st.info("Purchase contribution is not available in this report.")

st.divider()
section_header("Detailed segment comparison", "Use the table to compare the exact values behind the charts.")

display_df = df.copy()
for col in ["Conversion_Rate", "Session_Share", "Purchase_Contribution"]:
    if col in display_df.columns:
        display_df[col] = display_df[col].map(lambda x: f"{x:.2f}%")

st.dataframe(display_df, use_container_width=True, hide_index=True)

if selected == "Traffic source codes":
    insight(
        "<b>Interpretation note:</b> TrafficType contains numeric dataset codes. "
        "They are not verified names of advertising platforms or marketing channels."
    )

st.divider()
section_header("Statistical evidence", "The tests help distinguish observed differences from random variation.")

omnibus = load_csv(MARKETING / "statistical_tests" / "omnibus_tests.csv")
pairwise = load_csv(MARKETING / "statistical_tests" / "pairwise_tests.csv")

if not omnibus.empty:
    segment_name = {
        "Monthly performance": "Month",
        "Visitor type": "VisitorType",
        "Traffic source codes": "TrafficType",
        "Weekday vs weekend": "DayType",
        "Product-related activity": "ProductRelated",
        "Product-related duration": "ProductDuration",
        "Bounce rate": "BounceRates",
        "Exit rate": "ExitRates",
    }.get(selected)

    matched = omnibus[omnibus["Segment"] == segment_name] if segment_name else omnibus.iloc[0:0]

    if not matched.empty:
        row = matched.iloc[0]
        p_value = row.get("Holm_Adjusted_P_Value")
        cramers_v = row.get("Cramers_V")
        x, y = st.columns(2)
        x.metric("Holm-adjusted p-value", f"{p_value:.3g}" if p_value is not None else "N/A")
        y.metric("Cramér's V", f"{cramers_v:.3f}" if cramers_v is not None else "N/A")

        significant = bool(row.get("Statistically_Significant_After_Holm", False))
        if significant:
            insight(
                "<b>Statistical result:</b> This segment has a statistically significant "
                "association with purchase outcome after global Holm correction. "
                "Effect size should still be considered when judging practical importance."
            )
        else:
            insight(
                "<b>Statistical result:</b> This segment did not pass the global "
                "Holm-adjusted significance threshold."
            )

    if not pairwise.empty and segment_name:
        pair_segment = pairwise[pairwise["Segment"] == segment_name].copy()
        if not pair_segment.empty:
            with st.expander("View pairwise comparisons"):
                st.dataframe(pair_segment, use_container_width=True, hide_index=True)
    with st.expander("View all omnibus tests"):
        st.dataframe(omnibus, use_container_width=True, hide_index=True)
else:
    st.info("Statistical test reports are not available.")

insight(
    "<b>Important:</b> These are observational session-level patterns. "
    "They do not prove that a segment or behavior causes a purchase. "
    "The dataset does not identify unique customers or campaign spending."
)
