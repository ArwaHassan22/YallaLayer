
from pathlib import Path
import json

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
MARKETING = REPORTS / "marketing"
METRICS = REPORTS / "metrics"
MODEL_PATH = ROOT / "models" / "yallalayer_model.joblib"


NAVY = "#101D36"
BLUE = "#1769D2"
GOLD = "#E9A928"
INK = "#192943"
MUTED = "#718096"
BG = "#F4F7FC"


def apply_theme():
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Manrope:wght@400;500;600;700;800&display=swap');

        :root {{
            --navy: {NAVY};
            --blue: {BLUE};
            --gold: {GOLD};
            --ink: {INK};
            --muted: {MUTED};
            --surface: #FFFFFF;
            --background: {BG};
        }}

        html, body, [class*="css"] {{
            font-family: 'DM Sans', 'Manrope', sans-serif;
        }}

        .stApp {{
            background: var(--background);
        }}

        [data-testid="stHeader"] {{
            background: rgba(244,247,252,0.88);
        }}

        [data-testid="stSidebar"] {{
            background: linear-gradient(180deg, #101D36 0%, #172B4D 100%);
            border-right: 1px solid rgba(255,255,255,0.08);
        }}

        [data-testid="stSidebar"] * {{
            color: #F4F7FC;
        }}

        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {{
            color: #B8C5D9;
        }}

        /* Sidebar branding */
        .brand-lockup {{
            padding: 12px 4px 22px 4px;
        }}

        .brand-eyebrow {{
            color: #E9A928;
            font-size: 11px;
            font-weight: 800;
            letter-spacing: 3px;
            text-transform: uppercase;
        }}

        .brand-name {{
            color: white;
            font-size: 27px;
            font-weight: 800;
            letter-spacing: -1.2px;
            margin-top: 5px;
        }}

        .brand-sub {{
            color: #B8C5D9;
            font-size: 11px;
            margin-top: 2px;
        }}

        .sidebar-section-label {{
            color: #E9A928;
            font-size: 10px;
            font-weight: 800;
            letter-spacing: 2px;
            margin: 8px 0 12px 0;
        }}

        /* Sidebar navigation */
        [data-testid="stSidebar"] [data-testid="stPageLink"] {{
            background: rgba(255,255,255,0.045);
            border: 1px solid rgba(255,255,255,0.055);
            border-radius: 10px;
            margin: 5px 0;
            padding: 3px 8px;
            transition: background 0.2s ease;
        }}

        [data-testid="stSidebar"] [data-testid="stPageLink"]:hover {{
            background: rgba(255,255,255,0.13);
            border-color: rgba(233,169,40,0.45);
        }}

        [data-testid="stSidebar"] [data-testid="stPageLink"] p {{
            color: #F4F7FC !important;
            font-size: 13px;
            font-weight: 600;
        }}

        /* Hero section */
        .hero {{
            background: linear-gradient(115deg, #101D36 0%, #173D70 72%, #1769D2 100%);
            border-radius: 22px;
            padding: 34px 38px;
            color: white;
            margin: 8px 0 26px 0;
            box-shadow: 0 14px 34px rgba(16,29,54,0.12);
            position: relative;
            overflow: hidden;
        }}

        .hero:after {{
            content: "";
            position: absolute;
            width: 230px;
            height: 230px;
            right: -65px;
            top: -95px;
            border: 1px solid rgba(255,255,255,0.15);
            border-radius: 50%;
            box-shadow: 0 0 0 25px rgba(255,255,255,0.035),
                        0 0 0 55px rgba(255,255,255,0.025);
            pointer-events: none;
        }}

        .hero-kicker {{
            color: #F4C86A;
            text-transform: uppercase;
            letter-spacing: 2.5px;
            font-size: 10px;
            font-weight: 800;
        }}

        .hero-title {{
            font-size: clamp(28px, 4vw, 42px);
            line-height: 1.12;
            font-weight: 800;
            letter-spacing: -1.5px;
            margin: 12px 0;
        }}

        .hero-copy {{
            color: #D5E0F0;
            font-size: 14px;
            max-width: 650px;
            line-height: 1.7;
        }}

        /* Page headings */
        .page-kicker {{
            color: var(--blue);
            font-size: 10px;
            font-weight: 800;
            letter-spacing: 2.2px;
            text-transform: uppercase;
            margin-bottom: 5px;
        }}

        .page-title {{
            color: var(--navy);
            font-size: 32px;
            font-weight: 800;
            letter-spacing: -1px;
            margin: 0;
        }}

        .page-description {{
            color: var(--muted);
            font-size: 14px;
            margin-top: 7px;
            margin-bottom: 22px;
        }}

        .section-heading {{
            color: var(--navy);
            font-size: 19px;
            font-weight: 800;
            letter-spacing: -0.4px;
            margin: 18px 0 3px 0;
        }}

        .section-caption {{
            color: var(--muted);
            font-size: 12px;
            margin-bottom: 14px;
        }}

        /* KPI cards */
        [data-testid="stMetric"] {{
            background: white;
            border: 1px solid #E7ECF4;
            border-radius: 16px;
            padding: 18px 20px;
            box-shadow: 0 5px 18px rgba(16,29,54,0.035);
        }}

        [data-testid="stMetricLabel"] {{
            color: #718096;
            font-weight: 600;
            font-size: 12px;
        }}

        [data-testid="stMetricValue"] {{
            color: #101D36;
            font-weight: 800;
        }}

        /* Charts and tables */
        [data-testid="stPlotlyChart"] {{
            background: white;
            border: 1px solid #E7ECF4;
            border-radius: 16px;
            padding: 8px;
        }}

        [data-testid="stDataFrame"] {{
            border: 1px solid #E7ECF4;
            border-radius: 14px;
            overflow: hidden;
        }}

        /* Insight cards */
        .insight-card {{
            background: white;
            border: 1px solid #E7ECF4;
            border-left: 4px solid {GOLD};
            border-radius: 12px;
            padding: 16px 18px;
            margin: 8px 0;
            color: #334155;
            font-size: 13px;
            line-height: 1.65;
        }}

        .gold-label {{
            color: {GOLD};
            font-weight: 800;
        }}

        /* Forms */
        div[data-testid="stForm"] {{
            background: white;
            border: 1px solid #E7ECF4;
            border-radius: 16px;
            padding: 22px;
        }}

        /* Buttons */
        .stButton > button,
        .stFormSubmitButton > button {{
            background: {BLUE};
            color: white;
            border: 0;
            border-radius: 10px;
            padding: 9px 20px;
            font-weight: 700;
        }}

        .stButton > button:hover,
        .stFormSubmitButton > button:hover {{
            background: {NAVY};
            color: white;
            border: 0;
        }}

        hr {{
            border-color: #E4EAF2;
        }}

        footer {{
            visibility: hidden;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def sidebar_brand():
    with st.sidebar:
        st.markdown(
            """
            <div class="brand-lockup">
                <div class="brand-eyebrow">YALLA START ACADEMY</div>
                <div class="brand-name">
                    YallaLayer<span style="color:#E9A928">.</span>
                </div>
                <div class="brand-sub">
                    MARKETING INTELLIGENCE PLATFORM
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.divider()

        st.markdown(
            '<div class="sidebar-section-label">EXPLORE PLATFORM</div>',
            unsafe_allow_html=True,
        )

        st.page_link(
            "app.py",
            label="Executive Overview",
            icon=":material/dashboard:",
        )

        st.page_link(
            "pages/2_Marketing_Intelligence.py",
            label="Marketing Intelligence",
            icon=":material/insights:",
        )

        st.page_link(
            "pages/3_Purchase_Prediction.py",
            label="Purchase Prediction",
            icon=":material/query_stats:",
        )

        st.page_link(
            "pages/4_Model_Explainability.py",
            label="Model Explainability",
            icon=":material/analytics:",
        )

        st.page_link(
            "pages/5_About_and_Methodology.py",
            label="About & Methodology",
            icon=":material/info:",
        )

        st.divider()

        st.caption("A smarter view of e-commerce performance")
        st.caption("BUILT AS A YSA PROJECT")


def page_header(kicker, title, description):
    st.markdown(
        f"""
        <div class="page-kicker">{kicker}</div>
        <div class="page-title">{title}</div>
        <div class="page-description">{description}</div>
        """,
        unsafe_allow_html=True,
    )


def section_header(title, caption=None):
    st.markdown(
        f'<div class="section-heading">{title}</div>',
        unsafe_allow_html=True,
    )

    if caption:
        st.markdown(
            f'<div class="section-caption">{caption}</div>',
            unsafe_allow_html=True,
        )


def insight(text):
    st.markdown(
        f'<div class="insight-card">{text}</div>',
        unsafe_allow_html=True,
    )


@st.cache_data
def load_json(filename):
    path = filename if isinstance(filename, Path) else REPORTS / filename

    if not path.exists():
        return {}

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


@st.cache_data
def load_csv(path):
    path = Path(path)

    if not path.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(path)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError):
        return pd.DataFrame()


def find_metric(data, keys):
    keys = {key.lower() for key in keys}

    if isinstance(data, dict):
        for key, value in data.items():
            if key.lower() in keys and not isinstance(value, (dict, list)):
                return value

        for value in data.values():
            result = find_metric(value, keys)

            if result is not None:
                return result

    if isinstance(data, list):
        for item in data:
            result = find_metric(item, keys)

            if result is not None:
                return result

    return None


def percent(value):
    if value is None:
        return "N/A"

    value = float(value)

    if abs(value) <= 1:
        value *= 100

    return f"{value:.2f}%"


def number(value):
    if value is None:
        return "N/A"

    return f"{float(value):,.0f}"


def plot_layout(fig, height=360):
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=18, r=18, t=45, b=18),
        font=dict(family="DM Sans", color=INK),
        title_font=dict(size=15, color=NAVY),
        paper_bgcolor="white",
        plot_bgcolor="white",
        legend_title_text="",
    )

    fig.update_xaxes(
        showgrid=False,
        linecolor="#E7ECF4",
    )

    fig.update_yaxes(
        gridcolor="#EDF1F6",
        zeroline=False,
    )

    return fig
