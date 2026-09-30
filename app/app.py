from pathlib import Path
from html import escape
import base64
import mimetypes

import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
FAVICON_PATH = APP_DIR / "favicon.png"
ICON_PATH = APP_DIR / "icon.png"

st.set_page_config(
    page_title="E-Learning Behavioral Segmentation",
    page_icon=str(FAVICON_PATH) if FAVICON_PATH.exists() else "🎓",
    layout="wide",
    initial_sidebar_state="collapsed",
)

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "student_behaviour_clustered.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "kmodes_model.pkl"
METADATA_PATH = PROJECT_ROOT / "models" / "model_metadata.pkl"

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_resource
def load_metadata():
    return joblib.load(METADATA_PATH)

def image_data_uri(path):
    if not path.exists():
        return ""
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{encoded}"

def render_html(content):
    st.html(content)

try:
    df = load_data()
    model = load_model()
    metadata = load_metadata()
except FileNotFoundError as error:
    st.error("Required project files were not found.")
    st.code(str(error))
    st.stop()

BEHAVIOUR_FEATURES = metadata["features"]
CLUSTER_NAMES = {int(key): value for key, value in metadata["cluster_names"].items()}
N_CLUSTERS = int(metadata["n_clusters"])

SEGMENT_META = {
    "Highly Engaged Consistent Learners": {
        "icon": "✦",
        "accent": "#64D2FF",
        "accent_2": "#5E5CE6",
        "eyebrow": "High engagement",
        "description": "Frequent platform activity, consistent learning patterns, diverse resource usage and strong assessment engagement.",
    },
    "Steady Assessment-Engaged Learners": {
        "icon": "◉",
        "accent": "#30D158",
        "accent_2": "#00C7BE",
        "eyebrow": "Steady progress",
        "description": "Moderate, stable participation with strong assessment engagement and generally timely submissions.",
    },
    "Irregular Late-Submission Learners": {
        "icon": "◒",
        "accent": "#FF9F0A",
        "accent_2": "#FF6B00",
        "eyebrow": "Needs consistency",
        "description": "Inconsistent learning activity, noticeable inactivity periods and a tendency toward late submissions.",
    },
    "Low-Engagement Inactive Learners": {
        "icon": "◇",
        "accent": "#FF453A",
        "accent_2": "#BF5AF2",
        "eyebrow": "At-risk pattern",
        "description": "Low activity, infrequent platform usage, limited resource diversity and substantial inactivity.",
    },
}

FEATURE_LABELS = {
    "Activity_Level": "Activity Level",
    "Learning_Frequency": "Learning Frequency",
    "Learning_Consistency": "Learning Consistency",
    "Content_Preference": "Content Preference",
    "Resource_Diversity": "Resource Diversity",
    "Assessment_Engagement": "Assessment Engagement",
    "Submission_Behaviour": "Submission Behaviour",
    "Inactivity_Pattern": "Inactivity Pattern",
}

FEATURE_OPTIONS = {
    "Activity_Level": ["Low", "Medium", "High"],
    "Learning_Frequency": ["Rare", "Moderate", "Frequent"],
    "Learning_Consistency": ["Irregular", "Moderately Regular", "Consistent"],
    "Content_Preference": ["Learning Content", "Discussion", "Assessment", "Navigation"],
    "Resource_Diversity": ["Narrow", "Moderate", "Diverse"],
    "Assessment_Engagement": ["Low", "Medium", "High"],
    "Submission_Behaviour": ["Early", "On-Time", "Late", "No Submission Data"],
    "Inactivity_Pattern": ["Low Inactivity", "Moderate Inactivity", "High Inactivity"],
}

render_html(
    f"""
    <style>
    html, body, [class*="css"] {{
        font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "SF Pro Text", "Helvetica Neue", Arial, sans-serif !important;
    }}

    html {{
        scroll-behavior: smooth;
    }}

    .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"],
    [data-testid="stMainBlockContainer"] {{
        color: inherit;
    }}

    .stApp::before {{
        content: "";
        position: fixed;
        inset: 0;
        pointer-events: none;
        z-index: 0;
        background:
            radial-gradient(circle at 10% 0%, rgba(0, 122, 255, .11), transparent 28rem),
            radial-gradient(circle at 92% 4%, rgba(175, 82, 222, .09), transparent 31rem),
            radial-gradient(circle at 52% 75%, rgba(100, 210, 255, .045), transparent 35rem);
    }}

    [data-testid="stHeader"] {{
        background: color-mix(in srgb, currentColor 3%, transparent) !important;
        border-bottom: 1px solid color-mix(in srgb, currentColor 8%, transparent);
        backdrop-filter: blur(28px) saturate(180%);
        -webkit-backdrop-filter: blur(28px) saturate(180%);
    }}

    [data-testid="stToolbar"] {{
        gap: .25rem;
        padding-right: .45rem;
    }}

    [data-testid="stToolbar"] button,
    [data-testid="stMainMenu"] button,
    [data-testid="stDeployButton"] button {{
        border-radius: 13px !important;
        border: 1px solid color-mix(in srgb, currentColor 10%, transparent) !important;
        background: color-mix(in srgb, currentColor 4%, transparent) !important;
        color: inherit !important;
        backdrop-filter: blur(20px) saturate(170%);
        -webkit-backdrop-filter: blur(20px) saturate(170%);
    }}

    [data-testid="stDeployButton"] button {{
        padding-left: .9rem !important;
        padding-right: .9rem !important;
        font-weight: 650 !important;
    }}

    [role="menu"] {{
        padding: .5rem !important;
        border-radius: 20px !important;
        overflow: hidden !important;
        backdrop-filter: blur(34px) saturate(190%) !important;
        -webkit-backdrop-filter: blur(34px) saturate(190%) !important;
    }}

    [role="menu"] [role="menuitem"],
    [role="menu"] button {{
        border-radius: 12px !important;
        transition: background 150ms ease, transform 150ms ease !important;
    }}

    [role="menu"] [role="menuitem"]:hover,
    [role="menu"] button:hover {{
        background: color-mix(in srgb, currentColor 8%, transparent) !important;
    }}

    [role="radiogroup"] {{
        padding: .24rem !important;
        border-radius: 15px !important;
    }}

    [role="radiogroup"] label {{
        border-radius: 11px !important;
    }}

    [role="radiogroup"] label:has(input:checked) {{
        box-shadow: 0 5px 14px rgba(0, 0, 0, .13) !important;
    }}

    .block-container {{
        position: relative;
        z-index: 1;
        max-width: 1180px;
        padding: calc(4.75rem + env(safe-area-inset-top)) 1.35rem 5.5rem;
    }}

    .app-identity {{
        display: flex;
        align-items: center;
        gap: 1rem;
        margin: 0 0 1.15rem;
        color: inherit;
    }}

    .app-icon-shell {{
        width: 70px;
        height: 70px;
        flex: 0 0 70px;
        padding: 4px;
        border-radius: 20px;
        background: color-mix(in srgb, currentColor 5%, transparent);
        border: 1px solid color-mix(in srgb, currentColor 11%, transparent);
        box-shadow: 0 18px 44px rgba(0, 0, 0, .15);
        backdrop-filter: blur(24px) saturate(180%);
        -webkit-backdrop-filter: blur(24px) saturate(180%);
    }}

    .app-icon-shell img {{
        width: 100%;
        height: 100%;
        object-fit: cover;
        border-radius: 16px;
        display: block;
    }}

    .app-name {{
        font-size: clamp(1.42rem, 3vw, 1.92rem);
        line-height: 1.05;
        letter-spacing: -.045em;
        font-weight: 760;
        color: inherit;
    }}

    .app-subtitle {{
        margin-top: .35rem;
        font-size: .86rem;
        font-weight: 520;
        opacity: .58;
    }}

    .hero {{
        position: relative;
        overflow: hidden;
        border-radius: 34px;
        padding: 2.05rem;
        margin-bottom: .9rem;
        border: 1px solid color-mix(in srgb, currentColor 12%, transparent);
        background:
            linear-gradient(145deg, color-mix(in srgb, currentColor 7%, transparent), color-mix(in srgb, currentColor 3%, transparent)),
            linear-gradient(120deg, rgba(0, 122, 255, .07), rgba(175, 82, 222, .055));
        box-shadow: 0 28px 76px rgba(0, 0, 0, .14);
        backdrop-filter: blur(36px) saturate(185%);
        -webkit-backdrop-filter: blur(36px) saturate(185%);
        color: inherit;
    }}

    .hero::before {{
        content: "";
        position: absolute;
        width: 310px;
        height: 310px;
        border-radius: 50%;
        top: -175px;
        right: -70px;
        background: radial-gradient(circle, rgba(100, 210, 255, .26), rgba(175, 82, 222, .08) 45%, transparent 72%);
        filter: blur(10px);
    }}

    .hero-kicker {{
        display: inline-flex;
        align-items: center;
        gap: .42rem;
        padding: .45rem .72rem;
        border-radius: 999px;
        background: color-mix(in srgb, currentColor 4%, transparent);
        border: 1px solid color-mix(in srgb, currentColor 9%, transparent);
        font-size: .75rem;
        font-weight: 680;
        letter-spacing: .018em;
        opacity: .72;
    }}

    .hero-title {{
        position: relative;
        max-width: 900px;
        margin: 1.02rem 0 .68rem;
        font-size: clamp(2.45rem, 5.7vw, 4.8rem);
        line-height: .96;
        letter-spacing: -.064em;
        font-weight: 790;
        color: inherit;
    }}

    .hero-copy {{
        position: relative;
        max-width: 760px;
        margin: 0;
        font-size: clamp(.96rem, 1.45vw, 1.08rem);
        line-height: 1.6;
        letter-spacing: -.012em;
        opacity: .62;
    }}

    .hero-pills {{
        position: relative;
        display: flex;
        flex-wrap: wrap;
        gap: .52rem;
        margin-top: 1.25rem;
    }}

    .hero-pill {{
        display: inline-flex;
        align-items: center;
        gap: .38rem;
        padding: .5rem .7rem;
        border-radius: 999px;
        background: color-mix(in srgb, currentColor 4%, transparent);
        border: 1px solid color-mix(in srgb, currentColor 8%, transparent);
        font-size: .76rem;
        font-weight: 620;
        opacity: .72;
    }}

    .metric-grid {{
        display: grid;
        grid-template-columns: repeat(4, minmax(0,1fr));
        gap: .75rem;
        margin: .8rem 0 1.25rem;
    }}

    .metric-card {{
        min-height: 128px;
        padding: 1rem;
        border-radius: 25px;
        background: linear-gradient(145deg, color-mix(in srgb, currentColor 6%, transparent), color-mix(in srgb, currentColor 3%, transparent));
        border: 1px solid color-mix(in srgb, currentColor 10%, transparent);
        box-shadow: 0 18px 48px rgba(0, 0, 0, .12);
        backdrop-filter: blur(30px) saturate(175%);
        -webkit-backdrop-filter: blur(30px) saturate(175%);
        color: inherit;
    }}

    .metric-icon {{
        width: 35px;
        height: 35px;
        display: grid;
        place-items: center;
        margin-bottom: .82rem;
        border-radius: 12px;
        background: linear-gradient(145deg, rgba(100, 210, 255, .18), rgba(175, 82, 222, .12));
        border: 1px solid color-mix(in srgb, currentColor 9%, transparent);
        font-size: .95rem;
    }}

    .metric-value {{
        font-size: clamp(1.52rem, 3vw, 2.08rem);
        font-weight: 750;
        letter-spacing: -.045em;
        line-height: 1;
        color: inherit;
    }}

    .metric-label {{
        margin-top: .42rem;
        font-size: .78rem;
        font-weight: 540;
        opacity: .58;
    }}

    .section-head {{
        margin: .22rem 0 .95rem;
        color: inherit;
    }}

    .section-kicker {{
        margin-bottom: .34rem;
        color: #007AFF;
        font-size: .73rem;
        font-weight: 730;
        letter-spacing: .08em;
        text-transform: uppercase;
    }}

    .section-title {{
        margin: 0;
        font-size: clamp(1.55rem, 3vw, 2.25rem);
        font-weight: 750;
        line-height: 1.08;
        letter-spacing: -.047em;
        color: inherit;
    }}

    .section-copy {{
        margin: .46rem 0 0;
        max-width: 760px;
        font-size: .92rem;
        line-height: 1.55;
        opacity: .62;
    }}

    .glass-card,
    .segment-hero,
    .result-card {{
        background: linear-gradient(145deg, color-mix(in srgb, currentColor 6%, transparent), color-mix(in srgb, currentColor 3%, transparent));
        border: 1px solid color-mix(in srgb, currentColor 10%, transparent);
        box-shadow: 0 22px 54px rgba(0, 0, 0, .12);
        backdrop-filter: blur(30px) saturate(175%);
        -webkit-backdrop-filter: blur(30px) saturate(175%);
        color: inherit;
    }}

    .glass-card {{
        padding: 1.05rem;
        border-radius: 27px;
    }}

    .distribution-stack {{
        display: grid;
        gap: .66rem;
    }}

    .distribution-row {{
        padding: .92rem .96rem;
        border-radius: 20px;
        background: color-mix(in srgb, currentColor 3.5%, transparent);
        border: 1px solid color-mix(in srgb, currentColor 7%, transparent);
        color: inherit;
    }}

    .distribution-top {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1rem;
    }}

    .distribution-name {{
        font-size: .88rem;
        font-weight: 650;
        letter-spacing: -.012em;
        color: inherit;
    }}

    .distribution-number {{
        font-size: .76rem;
        white-space: nowrap;
        opacity: .62;
    }}

    .progress-track {{
        height: 7px;
        overflow: hidden;
        margin-top: .72rem;
        border-radius: 999px;
        background: color-mix(in srgb, currentColor 8%, transparent);
    }}

    .progress-fill {{
        height: 100%;
        border-radius: inherit;
        box-shadow: 0 0 18px currentColor;
    }}

    .segment-hero {{
        padding: 1.25rem;
        border-radius: 28px;
    }}

    .segment-topline {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1rem;
    }}

    .segment-icon {{
        width: 48px;
        height: 48px;
        display: grid;
        place-items: center;
        border-radius: 16px;
        border: 1px solid color-mix(in srgb, currentColor 10%, transparent);
        box-shadow: 0 12px 32px rgba(0, 0, 0, .12);
        font-size: 1.15rem;
    }}

    .segment-chip {{
        padding: .42rem .64rem;
        border-radius: 999px;
        background: color-mix(in srgb, currentColor 4%, transparent);
        border: 1px solid color-mix(in srgb, currentColor 8%, transparent);
        font-size: .73rem;
        font-weight: 650;
        opacity: .66;
    }}

    .segment-title {{
        margin: 1rem 0 .42rem;
        font-size: clamp(1.35rem, 3vw, 2rem);
        font-weight: 740;
        letter-spacing: -.045em;
        color: inherit;
    }}

    .segment-description {{
        margin: 0;
        font-size: .91rem;
        line-height: 1.56;
        opacity: .62;
    }}

    .profile-grid {{
        display: grid;
        grid-template-columns: repeat(4, minmax(0,1fr));
        gap: .66rem;
        margin-top: .78rem;
    }}

    .profile-item {{
        min-height: 92px;
        padding: .88rem;
        border-radius: 20px;
        background: color-mix(in srgb, currentColor 3.5%, transparent);
        border: 1px solid color-mix(in srgb, currentColor 7%, transparent);
        color: inherit;
    }}

    .profile-label {{
        margin-bottom: .4rem;
        font-size: .70rem;
        line-height: 1.25;
        opacity: .42;
    }}

    .profile-value {{
        font-size: .89rem;
        font-weight: 640;
        line-height: 1.3;
        letter-spacing: -.014em;
        color: inherit;
    }}

    .result-card {{
        position: relative;
        overflow: hidden;
        margin-top: 1rem;
        padding: 1.3rem;
        border-radius: 28px;
    }}

    .result-card::after {{
        content: "";
        position: absolute;
        width: 185px;
        height: 185px;
        top: -82px;
        right: -56px;
        border-radius: 50%;
        background: radial-gradient(circle, var(--result-glow), transparent 71%);
        filter: blur(9px);
        pointer-events: none;
    }}

    .result-label {{
        font-size: .72rem;
        font-weight: 710;
        text-transform: uppercase;
        letter-spacing: .08em;
        opacity: .42;
    }}

    .result-title {{
        position: relative;
        max-width: 780px;
        margin: .52rem 0;
        font-size: clamp(1.5rem,4vw,2.32rem);
        font-weight: 770;
        line-height: 1.05;
        letter-spacing: -.048em;
        color: inherit;
    }}

    .result-copy {{
        position: relative;
        max-width: 760px;
        font-size: .91rem;
        line-height: 1.55;
        opacity: .62;
    }}

    [data-testid="stTabs"] [data-baseweb="tab-list"] {{
        gap: .3rem;
        overflow-x: auto;
        margin-bottom: 1.15rem;
        padding: .32rem;
        border-radius: 18px;
        background: color-mix(in srgb, currentColor 3.5%, transparent);
        border: 1px solid color-mix(in srgb, currentColor 8%, transparent);
        backdrop-filter: blur(25px) saturate(175%);
        -webkit-backdrop-filter: blur(25px) saturate(175%);
        scrollbar-width: none;
    }}

    [data-testid="stTabs"] [data-baseweb="tab-list"]::-webkit-scrollbar {{
        display: none;
    }}

    [data-testid="stTabs"] button[role="tab"] {{
        flex: 0 0 auto;
        height: auto;
        padding: .6rem .85rem;
        border-radius: 13px;
        border: 0;
        font-size: .84rem;
        font-weight: 640;
        opacity: .58;
        color: inherit;
    }}

    [data-testid="stTabs"] button[role="tab"][aria-selected="true"] {{
        opacity: 1;
        background: color-mix(in srgb, currentColor 8%, transparent);
        box-shadow: 0 7px 20px rgba(0, 0, 0, .10);
    }}

    [data-testid="stTabs"] [data-baseweb="tab-highlight"] {{
        display: none;
    }}

    [data-baseweb="select"] > div {{
        min-height: 48px;
        border-radius: 16px !important;
    }}

    [data-baseweb="popover"] [role="listbox"] {{
        border-radius: 18px !important;
        backdrop-filter: blur(30px) saturate(185%) !important;
        -webkit-backdrop-filter: blur(30px) saturate(185%) !important;
    }}

    [data-testid="stForm"] {{
        padding: 1.12rem;
        border-radius: 28px;
        background: linear-gradient(145deg, color-mix(in srgb, currentColor 5%, transparent), color-mix(in srgb, currentColor 2.5%, transparent));
        border: 1px solid color-mix(in srgb, currentColor 9%, transparent);
        box-shadow: 0 20px 52px rgba(0, 0, 0, .10);
        backdrop-filter: blur(30px) saturate(175%);
        -webkit-backdrop-filter: blur(30px) saturate(175%);
    }}

    [data-testid="stForm"] label p {{
        font-size: .78rem !important;
        font-weight: 610 !important;
    }}

    .stButton > button,
    [data-testid="stFormSubmitButton"] > button {{
        width: 100%;
        min-height: 48px;
        border: 0 !important;
        border-radius: 16px !important;
        background: linear-gradient(135deg, #007AFF, #5E5CE6) !important;
        color: white !important;
        font-weight: 700 !important;
        box-shadow: 0 13px 30px rgba(0, 122, 255, .20) !important;
        transition: transform 150ms ease, filter 150ms ease;
    }}

    .stButton > button:hover,
    [data-testid="stFormSubmitButton"] > button:hover {{
        transform: translateY(-1px);
        filter: brightness(1.055);
    }}

    .st-key-outcome_desktop_chart {{
        display: block;
    }}

    .st-key-outcome_mobile_chart {{
        display: none;
    }}

    .st-key-outcome_desktop_chart [data-testid="stPlotlyChart"],
    .st-key-outcome_mobile_chart [data-testid="stPlotlyChart"] {{
        width: 100%;
        overflow: hidden;
        border-radius: 24px;
    }}

    .st-key-outcome_desktop_chart [data-testid="stPlotlyChart"] > div,
    .st-key-outcome_mobile_chart [data-testid="stPlotlyChart"] > div {{
        width: 100% !important;
    }}

    [data-testid="stDataFrame"] {{
        overflow: hidden;
        border-radius: 20px;
    }}

    [data-testid="stExpander"] {{
        overflow: hidden;
        border-radius: 21px !important;
    }}

    [data-testid="stAlert"] {{
        border-radius: 18px;
    }}

    hr {{
        margin: 1.5rem 0 !important;
        opacity: .12;
    }}

    @media (max-width: 900px) {{
        .block-container {{
            padding: calc(4.55rem + env(safe-area-inset-top)) 1rem 4.5rem;
        }}

        .metric-grid {{
            grid-template-columns: repeat(2,minmax(0,1fr));
        }}

        .profile-grid {{
            grid-template-columns: repeat(2,minmax(0,1fr));
        }}

        .hero {{
            padding: 1.55rem;
            border-radius: 29px;
        }}
    }}

    @media (max-width: 680px) {{
        .block-container {{
            padding: calc(4.2rem + env(safe-area-inset-top)) .78rem 3.8rem;
        }}

        [data-testid="stHeader"] {{
            height: 3rem;
        }}

        [data-testid="stToolbar"] {{
            padding-right: .2rem;
        }}

        [data-testid="stDeployButton"] button {{
            min-height: 34px !important;
            padding: 0 .65rem !important;
            border-radius: 11px !important;
            font-size: .78rem !important;
        }}

        .app-identity {{
            gap: .72rem;
            margin: 0 0 1rem;
            align-items: center;
        }}

        .app-icon-shell {{
            width: 54px;
            height: 54px;
            flex-basis: 54px;
            border-radius: 16px;
            padding: 3px;
        }}

        .app-icon-shell img {{
            border-radius: 13px;
        }}

        .app-name {{
            font-size: 1.18rem;
            line-height: 1.12;
        }}

        .app-subtitle {{
            font-size: .72rem;
            line-height: 1.25;
        }}

        .hero {{
            padding: 1.25rem 1.06rem;
            border-radius: 24px;
        }}

        .hero-title {{
            font-size: clamp(2rem, 11.5vw, 2.65rem);
            letter-spacing: -.061em;
        }}

        .hero-copy {{
            font-size: .88rem;
            line-height: 1.58;
        }}

        .hero-pills {{
            gap: .4rem;
        }}

        .hero-pill {{
            padding: .43rem .56rem;
            font-size: .68rem;
        }}

        .metric-grid {{
            gap: .56rem;
        }}

        .metric-card {{
            min-height: 108px;
            padding: .8rem;
            border-radius: 19px;
        }}

        .metric-icon {{
            width: 30px;
            height: 30px;
            margin-bottom: .62rem;
            border-radius: 10px;
        }}

        .metric-value {{
            font-size: 1.42rem;
        }}

        .metric-label {{
            font-size: .72rem;
        }}

        .profile-grid {{
            grid-template-columns: 1fr 1fr;
            gap: .5rem;
        }}

        .profile-item {{
            min-height: 80px;
            padding: .72rem;
            border-radius: 16px;
        }}

        .profile-value {{
            font-size: .80rem;
            overflow-wrap: anywhere;
        }}

        [data-testid="stHorizontalBlock"] {{
            flex-direction: column !important;
            gap: .15rem !important;
        }}

        [data-testid="stHorizontalBlock"] > div {{
            width: 100% !important;
            flex: 1 1 100% !important;
        }}

        [data-testid="stTabs"] [data-baseweb="tab-list"] {{
            position: sticky;
            top: 3.1rem;
            z-index: 7;
            padding: .27rem;
            border-radius: 16px;
            backdrop-filter: blur(28px) saturate(185%);
            -webkit-backdrop-filter: blur(28px) saturate(185%);
        }}

        [data-testid="stTabs"] button[role="tab"] {{
            padding: .52rem .68rem;
            font-size: .76rem;
        }}

        .st-key-outcome_desktop_chart {{
            display: none !important;
        }}

        .st-key-outcome_mobile_chart {{
            display: block !important;
            width: 100% !important;
            max-width: 100% !important;
        }}

        .st-key-outcome_mobile_chart [data-testid="stPlotlyChart"] {{
            width: 100% !important;
            max-width: 100% !important;
            margin: 0 !important;
            border-radius: 18px;
        }}

        [data-testid="stForm"] {{
            padding: .84rem;
            border-radius: 21px;
        }}

        .segment-hero,
        .result-card {{
            padding: 1rem;
            border-radius: 22px;
        }}

        .distribution-row {{
            padding: .76rem;
            border-radius: 16px;
        }}

        .distribution-top {{
            align-items: flex-start;
            flex-direction: column;
            gap: .3rem;
        }}

        .distribution-name {{
            font-size: .80rem;
            line-height: 1.3;
        }}

        .distribution-number {{
            font-size: .68rem;
        }}

        [role="menu"] {{
            max-width: min(94vw, 330px) !important;
            border-radius: 19px !important;
        }}
    }}

    @media (max-width: 390px) {{
        .profile-grid {{
            grid-template-columns: 1fr;
        }}

        .hero-title {{
            font-size: 2rem;
        }}

        .app-subtitle {{
            display: none;
        }}

        .app-name {{
            font-size: 1.1rem;
        }}
    }}
    </style>
    """,
)

total_profiles = len(df)
unique_students = df["id_student"].nunique()
modules = df["code_module"].nunique()

icon_source = image_data_uri(ICON_PATH if ICON_PATH.exists() else FAVICON_PATH)
if icon_source:
    icon_html = f'<div class="app-icon-shell"><img src="{icon_source}" alt="App icon"></div>'
else:
    icon_html = '<div class="app-icon-shell" style="display:grid;place-items:center;font-size:1.7rem;">🎓</div>'

render_html(
    f"""
    <div class="app-identity">
        {icon_html}
        <div>
            <div class="app-name">E-Learning Behavioral Segmentation</div>
            <div class="app-subtitle">Student behavior intelligence · K-Modes clustering</div>
        </div>
    </div>
    """,
)

render_html(
    f"""
    <section class="hero">
        <span class="hero-kicker">✦ Unsupervised learning · OULAD</span>
        <h1 class="hero-title">Student behavior,<br>made visible.</h1>
        <p class="hero-copy">
            Discover how learners engage, stay consistent, submit assessments and drift into inactivity.
            The engine transforms learning interaction data into clear behavioral segments without using academic outcomes to create the clusters.
        </p>
        <div class="hero-pills">
            <span class="hero-pill">◉ {total_profiles:,} profiles</span>
            <span class="hero-pill">⌁ {N_CLUSTERS} segments</span>
            <span class="hero-pill">◇ 8 behavioral signals</span>
            <span class="hero-pill">✦ K-Modes</span>
        </div>
    </section>
    """,
)

render_html(
    f"""
    <div class="metric-grid">
        <div class="metric-card">
            <div class="metric-icon">◉</div>
            <div class="metric-value">{total_profiles:,}</div>
            <div class="metric-label">Behavioral profiles</div>
        </div>
        <div class="metric-card">
            <div class="metric-icon">◎</div>
            <div class="metric-value">{unique_students:,}</div>
            <div class="metric-label">Unique students</div>
        </div>
        <div class="metric-card">
            <div class="metric-icon">✦</div>
            <div class="metric-value">{N_CLUSTERS}</div>
            <div class="metric-label">Behavioral segments</div>
        </div>
        <div class="metric-card">
            <div class="metric-icon">⌁</div>
            <div class="metric-value">{modules}</div>
            <div class="metric-label">Course modules</div>
        </div>
    </div>
    """,
)

overview_tab, segments_tab, outcomes_tab, predict_tab, about_tab = st.tabs(
    ["Overview", "Segments", "Outcomes", "Predict", "About"]
)

cluster_distribution = (
    df["Cluster_Name"]
    .value_counts()
    .rename_axis("Behavioral Segment")
    .reset_index(name="Students")
)

cluster_distribution["Percentage"] = (
    cluster_distribution["Students"] / cluster_distribution["Students"].sum() * 100
).round(2)

with overview_tab:
    render_html(
        """
        <div class="section-head">
            <div class="section-kicker">Population view</div>
            <h2 class="section-title">Behavioral segment distribution</h2>
            <p class="section-copy">See how the complete student-course population is distributed across the four discovered learning patterns.</p>
        </div>
        """,
    )

    distribution_html = '<div class="glass-card"><div class="distribution-stack">'
    for _, row in cluster_distribution.iterrows():
        name = str(row["Behavioral Segment"])
        count = int(row["Students"])
        percentage = float(row["Percentage"])
        meta = SEGMENT_META.get(name, {})
        accent = meta.get("accent", "#64D2FF")
        accent_2 = meta.get("accent_2", "#5E5CE6")
        distribution_html += f"""
        <div class="distribution-row">
            <div class="distribution-top">
                <div class="distribution-name">{escape(name)}</div>
                <div class="distribution-number">{count:,} · {percentage:.2f}%</div>
            </div>
            <div class="progress-track">
                <div class="progress-fill" style="width:{percentage}%; background:linear-gradient(90deg,{accent},{accent_2}); color:{accent};"></div>
            </div>
        </div>
        """
    distribution_html += "</div></div>"
    render_html(distribution_html)

    render_html("<div style='height:.82rem'></div>")

    left, right = st.columns([1.35, 1])

    with left:
        render_html(
            """
            <div class="section-head">
                <div class="section-kicker">What the model sees</div>
                <h3 class="section-title" style="font-size:1.35rem;">Eight behavioral signals</h3>
            </div>
            """,
        )
        feature_cards = '<div class="profile-grid">'
        for feature in BEHAVIOUR_FEATURES:
            feature_cards += f"""
            <div class="profile-item">
                <div class="profile-label">Signal</div>
                <div class="profile-value">{escape(FEATURE_LABELS.get(feature, feature))}</div>
            </div>
            """
        feature_cards += "</div>"
        render_html(feature_cards)

    with right:
        render_html(
            """
            <div class="section-head">
                <div class="section-kicker">Model snapshot</div>
                <h3 class="section-title" style="font-size:1.35rem;">K-Modes clustering</h3>
                <p class="section-copy">Categorical behavior patterns are grouped by mode similarity rather than numerical distance.</p>
            </div>
            """,
        )
        render_html(
            f"""
            <div class="glass-card">
                <div class="profile-grid model-grid">
                    <div class="profile-item">
                        <div class="profile-label">Algorithm</div>
                        <div class="profile-value">K-Modes</div>
                    </div>
                    <div class="profile-item">
                        <div class="profile-label">Clusters</div>
                        <div class="profile-value">{N_CLUSTERS}</div>
                    </div>
                    <div class="profile-item">
                        <div class="profile-label">Feature type</div>
                        <div class="profile-value">Categorical</div>
                    </div>
                    <div class="profile-item">
                        <div class="profile-label">Validation</div>
                        <div class="profile-value">External outcomes</div>
                    </div>
                </div>
            </div>
            """,
        )

with segments_tab:
    render_html(
        """
        <div class="section-head">
            <div class="section-kicker">Segment explorer</div>
            <h2 class="section-title">Inspect a behavioral archetype</h2>
            <p class="section-copy">Choose a segment to reveal its typical profile and modal category for every behavioral signal.</p>
        </div>
        """,
    )

    segment_names = sorted(df["Cluster_Name"].dropna().unique())
    selected_segment = st.selectbox(
        "Behavioral segment",
        segment_names,
        label_visibility="collapsed",
    )

    segment_data = df[df["Cluster_Name"] == selected_segment]
    cluster_id = int(segment_data["Cluster"].iloc[0])
    meta = SEGMENT_META.get(selected_segment, {})
    segment_icon = meta.get("icon", "✦")
    segment_accent = meta.get("accent", "#64D2FF")
    segment_accent_2 = meta.get("accent_2", "#5E5CE6")
    segment_eyebrow = meta.get("eyebrow", "Behavioral pattern")
    segment_description = meta.get(
        "description",
        "Behavioral segment identified through K-Modes clustering.",
    )

    render_html(
        f"""
        <div class="segment-hero">
            <div class="segment-topline">
                <div class="segment-icon" style="background:linear-gradient(145deg,{segment_accent}2A,{segment_accent_2}22);color:{segment_accent};">{segment_icon}</div>
                <div class="segment-chip">Cluster {cluster_id} · {len(segment_data):,} profiles</div>
            </div>
            <div class="segment-title">{escape(selected_segment)}</div>
            <div class="section-kicker" style="color:{segment_accent};margin-bottom:.45rem;">{escape(segment_eyebrow)}</div>
            <p class="segment-description">{escape(segment_description)}</p>
        </div>
        """,
    )

    cluster_mode = pd.DataFrame(model.cluster_centroids_, columns=BEHAVIOUR_FEATURES)
    selected_mode = cluster_mode.loc[cluster_id]

    render_html(
        """
        <div class="section-head" style="margin-top:1.25rem;">
            <div class="section-kicker">Typical profile</div>
            <h3 class="section-title" style="font-size:1.35rem;">Cluster mode</h3>
        </div>
        """,
    )

    profile_html = '<div class="profile-grid">'
    for feature, value in selected_mode.items():
        profile_html += f"""
        <div class="profile-item">
            <div class="profile-label">{escape(FEATURE_LABELS.get(feature, feature))}</div>
            <div class="profile-value">{escape(str(value))}</div>
        </div>
        """
    profile_html += "</div>"
    render_html(profile_html)

with outcomes_tab:
    render_html(
        """
        <div class="section-head">
            <div class="section-kicker">External validation</div>
            <h2 class="section-title">Academic outcomes by segment</h2>
            <p class="section-copy">Outcomes were not used to create the clusters. They are shown afterward to understand how the discovered behavior patterns relate to final results.</p>
        </div>
        """,
    )

    outcome_percentage = pd.crosstab(
        df["Cluster_Name"],
        df["final_result"],
        normalize="index",
    ).mul(100).round(2)

    outcome_order = ["Distinction", "Pass", "Fail", "Withdrawn"]
    outcome_percentage = outcome_percentage.reindex(columns=outcome_order, fill_value=0)

    segment_order = list(outcome_percentage.index)

    display_labels = {
        "Highly Engaged Consistent Learners": "Highly Engaged<br>Consistent<br>Learners",
        "Steady Assessment-Engaged Learners": "Steady Assessment-<br>Engaged<br>Learners",
        "Irregular Late-Submission Learners": "Irregular Late-<br>Submission<br>Learners",
        "Low-Engagement Inactive Learners": "Low-Engagement<br>Inactive<br>Learners",
    }

    chart_colors = {
        "Distinction": "#4F8EF7",
        "Pass": "#22B983",
        "Fail": "#E85D75",
        "Withdrawn": "#E7A63A",
    }

    fig = go.Figure()

    for outcome in outcome_order:
        values = outcome_percentage[outcome]

        fig.add_trace(
            go.Bar(
                name=outcome,
                x=[display_labels.get(name, name) for name in segment_order],
                y=values,
                text=[f"{value:.1f}%" for value in values],
                textposition="outside",
                textfont=dict(size=11),
                width=0.15,
                cliponaxis=False,
                marker=dict(
                    color=chart_colors[outcome],
                    line=dict(width=0),
                ),
                hovertemplate=(
                    "<b>%{x}</b><br>"
                    + outcome
                    + ": %{y:.2f}%<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        barmode="group",
        height=540,
        autosize=True,
        margin=dict(l=48, r=24, t=78, b=112),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        hovermode="closest",
        bargap=0.22,
        bargroupgap=0.08,
        uniformtext_minsize=9,
        uniformtext_mode="show",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.04,
            xanchor="center",
            x=0.5,
            title=None,
            font=dict(size=11),
        ),
        xaxis=dict(
            title=None,
            showgrid=False,
            zeroline=False,
            automargin=True,
            tickangle=0,
            tickfont=dict(size=11),
            ticklabelstandoff=10,
            fixedrange=True,
        ),
        yaxis=dict(
            title=dict(
                text="Share of segment (%)",
                font=dict(size=11),
            ),
            range=[0, 80],
            dtick=20,
            showgrid=True,
            gridcolor="rgba(128,128,128,0.16)",
            zeroline=False,
            automargin=True,
            tickfont=dict(size=10),
            fixedrange=True,
        ),
        font=dict(
            family='-apple-system, BlinkMacSystemFont, "SF Pro Text", "Helvetica Neue", Arial, sans-serif',
            size=12,
        ),
        hoverlabel=dict(
            font_size=12,
            font_family='-apple-system, BlinkMacSystemFont, "SF Pro Text", "Helvetica Neue", Arial, sans-serif',
        ),
    )

    with st.container(key="outcome_desktop_chart"):
        st.plotly_chart(
            fig,
            use_container_width=True,
            theme="streamlit",
            config={
                "displayModeBar": False,
                "displaylogo": False,
                "responsive": True,
                "scrollZoom": False,
                "doubleClick": False,
            },
        )

    mobile_segment_labels = {
        "Highly Engaged Consistent Learners": "Highly Engaged Consistent<br>Learners",
        "Steady Assessment-Engaged Learners": "Steady Assessment-Engaged<br>Learners",
        "Irregular Late-Submission Learners": "Irregular Late-Submission<br>Learners",
        "Low-Engagement Inactive Learners": "Low-Engagement Inactive<br>Learners",
    }

    mobile_fig = go.Figure()

    mobile_domains = [
        (0.735, 0.875),
        (0.505, 0.645),
        (0.275, 0.415),
        (0.045, 0.185),
    ]

    mobile_title_positions = [0.915, 0.685, 0.455, 0.225]

    for index, segment_name in enumerate(segment_order):
        values = [float(outcome_percentage.loc[segment_name, outcome]) for outcome in outcome_order]
        y0, y1 = mobile_domains[index]

        mobile_fig.add_trace(
            go.Pie(
                labels=outcome_order,
                values=values,
                hole=0.58,
                sort=False,
                direction="clockwise",
                marker=dict(
                    colors=[chart_colors[outcome] for outcome in outcome_order],
                    line=dict(color="rgba(255,255,255,0.10)", width=1),
                ),
                textinfo="percent",
                textposition="inside",
                textfont=dict(size=11),
                hovertemplate="<b>%{label}</b><br>%{value:.2f}%<extra></extra>",
                domain=dict(x=[0.16, 0.84], y=[y0, y1]),
                showlegend=index == 0,
                name=segment_name,
            )
        )

        mobile_fig.add_annotation(
            x=0.5,
            y=mobile_title_positions[index],
            xref="paper",
            yref="paper",
            text=f"<b>{mobile_segment_labels.get(segment_name, segment_name)}</b>",
            showarrow=False,
            align="center",
            xanchor="center",
            yanchor="middle",
            font=dict(size=12),
        )

    mobile_fig.update_layout(
        height=1260,
        margin=dict(l=6, r=6, t=96, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.015,
            xanchor="center",
            x=0.5,
            font=dict(size=10),
            traceorder="normal",
        ),
        font=dict(
            family='-apple-system, BlinkMacSystemFont, "SF Pro Text", "Helvetica Neue", Arial, sans-serif',
            size=11,
        ),
        hoverlabel=dict(
            font_size=12,
            font_family='-apple-system, BlinkMacSystemFont, "SF Pro Text", "Helvetica Neue", Arial, sans-serif',
        ),
    )

    with st.container(key="outcome_mobile_chart"):
        st.plotly_chart(
            mobile_fig,
            use_container_width=True,
            theme="streamlit",
            config={
                "displayModeBar": False,
                "displaylogo": False,
                "responsive": True,
                "scrollZoom": False,
                "doubleClick": False,
            },
        )

    render_html("<div style='height:.45rem'></div>")

    st.dataframe(
        outcome_percentage,
        use_container_width=True,
    )

    st.caption(
        "These differences are associations, not evidence that a behavioral segment causes a particular academic outcome."
    )

    render_html(
        """
        <div class="section-head" style="margin-top:1.2rem;">
            <div class="section-kicker">Focused view</div>
            <h3 class="section-title" style="font-size:1.35rem;">Inspect one segment</h3>
        </div>
        """,
    )

    outcome_segment = st.selectbox(
        "Outcome segment",
        sorted(df["Cluster_Name"].dropna().unique()),
        key="outcome_segment",
        label_visibility="collapsed",
    )

    outcome_segment_data = df[df["Cluster_Name"] == outcome_segment]
    selected_outcomes = (
        outcome_segment_data["final_result"]
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
        .rename_axis("Final Result")
        .reset_index(name="Percentage")
    )

    st.dataframe(
        selected_outcomes,
        hide_index=True,
        use_container_width=True,
    )

with predict_tab:
    render_html(
        """
        <div class="section-head">
            <div class="section-kicker">Live inference</div>
            <h2 class="section-title">Predict a behavioral segment</h2>
            <p class="section-copy">Enter an engineered categorical student profile. The saved K-Modes model assigns it to the closest learned behavioral segment.</p>
        </div>
        """,
    )

    with st.form("student_prediction_form"):
        left_col, right_col = st.columns(2)
        user_input = {}

        for index, feature in enumerate(BEHAVIOUR_FEATURES):
            target = left_col if index < 4 else right_col
            with target:
                user_input[feature] = st.selectbox(
                    FEATURE_LABELS.get(feature, feature),
                    FEATURE_OPTIONS[feature],
                    key=f"predict_{feature}",
                )

        submitted = st.form_submit_button(
            "Predict behavioral segment",
            use_container_width=True,
        )

    if submitted:
        input_df = pd.DataFrame(
            [[user_input[feature] for feature in BEHAVIOUR_FEATURES]],
            columns=BEHAVIOUR_FEATURES,
        )

        predicted_cluster = int(model.predict(input_df)[0])
        predicted_segment = CLUSTER_NAMES[predicted_cluster]
        predicted_meta = SEGMENT_META.get(predicted_segment, {})
        predicted_accent = predicted_meta.get("accent", "#64D2FF")
        predicted_description = predicted_meta.get(
            "description",
            "Behavioral segment identified by the trained K-Modes model.",
        )

        render_html(
            f"""
            <div class="result-card" style="--result-glow:{predicted_accent}66;">
                <div class="result-label">Predicted segment · Cluster {predicted_cluster}</div>
                <div class="result-title">{escape(predicted_segment)}</div>
                <div class="result-copy">{escape(predicted_description)}</div>
            </div>
            """,
        )

        prediction_profile = pd.DataFrame(
            {
                "Behavioral Signal": [
                    FEATURE_LABELS.get(feature, feature) for feature in BEHAVIOUR_FEATURES
                ],
                "Selected Value": [user_input[feature] for feature in BEHAVIOUR_FEATURES],
            }
        )

        render_html(
            """
            <div class="section-head" style="margin-top:1.1rem;">
                <div class="section-kicker">Input profile</div>
                <h3 class="section-title" style="font-size:1.35rem;">What the model received</h3>
            </div>
            """,
        )

        st.dataframe(
            prediction_profile,
            hide_index=True,
            use_container_width=True,
        )

    st.caption(
        "This tool predicts behavioral segment membership. It does not predict Pass, Fail, Distinction or Withdrawal."
    )

with about_tab:
    render_html(
        """
        <div class="section-head">
            <div class="section-kicker">Methodology</div>
            <h2 class="section-title">How the engine works</h2>
            <p class="section-copy">A compact view of the data, feature design and clustering logic behind the dashboard.</p>
        </div>
        """,
    )

    render_html(
        f"""
        <div class="glass-card">
            <div class="profile-grid about-grid">
                <div class="profile-item">
                    <div class="profile-label">Dataset</div>
                    <div class="profile-value">OULAD</div>
                </div>
                <div class="profile-item">
                    <div class="profile-label">Algorithm</div>
                    <div class="profile-value">K-Modes</div>
                </div>
                <div class="profile-item">
                    <div class="profile-label">Clusters</div>
                    <div class="profile-value">{N_CLUSTERS}</div>
                </div>
            </div>
        </div>
        """,
    )

    render_html("<div style='height:.9rem'></div>")

    with st.expander("Behavioral features", expanded=True):
        feature_table = pd.DataFrame(
            {
                "Feature": [FEATURE_LABELS.get(feature, feature) for feature in BEHAVIOUR_FEATURES],
                "Categories": [", ".join(FEATURE_OPTIONS[feature]) for feature in BEHAVIOUR_FEATURES],
            }
        )
        st.dataframe(feature_table, hide_index=True, use_container_width=True)

    with st.expander("Model selection"):
        st.markdown(
            "K-Modes was selected because the final behavioral features are categorical. Multiple values of K were compared using clustering cost, Hamming-distance silhouette analysis, cluster size and behavioral interpretability. The final model uses four clusters because it preserves meaningful behavioral separation while keeping each segment substantial enough to interpret."
        )

    with st.expander("Outcome interpretation"):
        st.markdown(
            "`final_result` is intentionally excluded from clustering. Pass, Fail, Distinction and Withdrawn are used only after clustering as external validation. This keeps the segmentation behavior-driven rather than outcome-driven."
        )

render_html(
    """
    <div style="text-align:center;padding:2.2rem 0 .2rem;color:var(--ios-faint);font-size:.72rem;">
        E-Learning Behavioral Segmentation · K-Modes · OULAD
    </div>
    """,
)
