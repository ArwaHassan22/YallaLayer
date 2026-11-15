
import pandas as pd
import plotly.express as px
import streamlit as st

from src.yallalayer.dashboard_utils import (
    MARKETING,
    apply_theme,
    sidebar_brand,
    page_header,
    section_header,
    insight,
    load_csv,
    load_json,
    plot_layout,
)


# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="YallaLayer | YSA",
    page_icon="Y",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==================================================
# GLOBAL DESIGN & NAVIGATION
# ==================================================

apply_theme()
sidebar_brand()


# ==================================================
# LOAD REPORTS
# ==================================================

marketing = load_json(
    MARKETING / "marketing_summary.json"
)

monthly = load_csv(
    MARKETING / "monthly_performance.csv"
)

visitors = load_csv(
    MARKETING / "visitor_type_performance.csv"
)


# ==================================================
# HERO SECTION
# ==================================================

st.markdown(
    """
    <div class="hero">
        <div class="hero-kicker">
            YALLA START ACADEMY · INTELLIGENCE SUITE
        </div>
        <div class="hero-title">
            Turn Customer Behavior<br>
            Into Marketing Wins.
        </div>
        <div class="hero-copy">
            Discover what drives conversions, uncover growth opportunities,
            and turn e-commerce data into smarter marketing decisions.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ==================================================
# EXECUTIVE OVERVIEW
# ==================================================

page_header(
    "YOUR BUSINESS AT A GLANCE",
    "Executive Overview",
    "A concise snapshot of e-commerce activity, marketing performance, and predictive insights.",
)


# ==================================================
# KEY PERFORMANCE INDICATORS
# ==================================================

sessions = int(
    marketing.get("total_sessions", 9864)
)

purchases = int(
    marketing.get("total_purchases", 1526)
)

conversion = float(
    marketing.get("overall_conversion_rate", 15.47)
)


k1, k2, k3, k4 = st.columns(4)

k1.metric(
    "Analyzed Sessions",
    f"{sessions:,}",
)

k2.metric(
    "Purchasing Sessions",
    f"{purchases:,}",
)

k3.metric(
    "Conversion Rate",
    f"{conversion:.2f}%",
)

k4.metric(
    "Test PR-AUC",
    "72.70%",
)


st.caption(
    "Marketing analysis uses the training split. "
    "Model metrics are reported on the held-out test split."
)


# ==================================================
# MONTHLY PERFORMANCE
# ==================================================

st.divider()

section_header(
    "Performance Over Time",
    "Track how conversion rates change across the months.",
)


required_monthly_columns = {
    "Month",
    "Conversion_Rate",
    "Sessions",
    "Purchases",
}


if (
    not monthly.empty
    and required_monthly_columns.issubset(monthly.columns)
):

    month_order = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "June",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]

    monthly = monthly.copy()

    monthly["Month"] = monthly["Month"].astype(str)

    monthly["Month"] = pd.Categorical(
        monthly["Month"],
        categories=month_order,
        ordered=True,
    )

    monthly = monthly.sort_values("Month")

    fig = px.area(
        monthly,
        x="Month",
        y="Conversion_Rate",
        markers=True,
        hover_data=[
            "Sessions",
            "Purchases",
        ],
        color_discrete_sequence=["#1769D2"],
        labels={
            "Conversion_Rate": "Conversion Rate (%)",
            "Month": "Month",
        },
    )

    fig.update_traces(
        fill="tozeroy",
        line=dict(width=3),
        fillcolor="rgba(23,105,210,0.10)",
        marker=dict(
            size=7,
            color="#E9A928",
        ),
    )

    st.plotly_chart(
        plot_layout(fig, 360),
        use_container_width=True,
    )

else:

    st.info(
        "Monthly performance report is unavailable "
        "or does not contain the required columns."
    )


# ==================================================
# VISITOR BEHAVIOR & MODEL PERFORMANCE
# ==================================================

left, right = st.columns(2)


with left:

    section_header(
        "Visitor Behavior",
        "Compare conversion rates across visitor types.",
    )

    required_visitor_columns = {
        "VisitorType",
        "Conversion_Rate",
        "Sessions",
        "Purchases",
    }

    if (
        not visitors.empty
        and required_visitor_columns.issubset(visitors.columns)
    ):

        fig_v = px.bar(
            visitors.sort_values(
                "Conversion_Rate",
                ascending=True,
            ),
            x="Conversion_Rate",
            y="VisitorType",
            orientation="h",
            color_discrete_sequence=["#1769D2"],
            hover_data=[
                "Sessions",
                "Purchases",
            ],
            labels={
                "Conversion_Rate": "Conversion Rate (%)",
                "VisitorType": "Visitor Type",
            },
        )

        st.plotly_chart(
            plot_layout(fig_v, 280),
            use_container_width=True,
        )

    else:

        st.info(
            "Visitor type report is unavailable."
        )


with right:

    section_header(
        "Model Performance",
        "Evaluation on the held-out test dataset.",
    )

    st.metric(
        "ROC-AUC",
        "92.29%",
    )

    st.metric(
        "Recall at Selected Threshold",
        "84.03%",
    )

    st.metric(
        "Precision at Selected Threshold",
        "51.52%",
    )

    st.caption(
        "Selected threshold: 0.1309 · Gradient Boosting"
    )


# ==================================================
# EXPLORE THE PLATFORM
# ==================================================

st.divider()

section_header(
    "Explore the Platform",
    "Go beyond the overview and discover what the data reveals.",
)


c1, c2, c3 = st.columns(3)


with c1:

    st.markdown(
        "**01 · Marketing Intelligence**"
    )

    st.write(
        "Explore customer-session segments, "
        "conversion patterns, and statistical evidence."
    )

    st.page_link(
        "pages/2_Marketing_Intelligence.py",
        label="Explore Marketing",
        icon=":material/arrow_forward:",
    )


with c2:

    st.markdown(
        "**02 · Purchase Prediction**"
    )

    st.write(
        "Estimate purchase intention using "
        "the trained machine learning model."
    )

    st.page_link(
        "pages/3_Purchase_Prediction.py",
        label="Try Prediction",
        icon=":material/arrow_forward:",
    )


with c3:

    st.markdown(
        "**03 · Model Explainability**"
    )

    st.write(
        "Understand which features influence "
        "the model's predictions."
    )

    st.page_link(
        "pages/4_Model_Explainability.py",
        label="Explore Explanations",
        icon=":material/arrow_forward:",
    )


# ==================================================
# ANALYTICAL NOTE
# ==================================================

insight(
    "<b>Analytical Note:</b> "
    "The dataset represents browsing sessions, not confirmed unique customers. "
    "Observed relationships indicate associations and should not be interpreted "
    "as proof of causation."
)


# ==================================================
# FOOTER
# ==================================================

st.markdown(
    """
    <div style="
        text-align:center;
        color:#8793A5;
        font-size:11px;
        padding:22px 0;
        letter-spacing:1px;
    ">
        YALLALAYER · A YSA PROJECT · MARKETING INTELLIGENCE
    </div>
    """,
    unsafe_allow_html=True,
)
