import os
from typing import Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import streamlit as st


FEATURE_COLUMNS = [
    "fgm_home", "fga_home", "fg_pct_home", "fg3m_home", "fg3a_home", "fg3_pct_home",
    "ftm_home", "fta_home", "ft_pct_home", "oreb_home", "dreb_home", "reb_home",
    "ast_home", "stl_home", "blk_home", "tov_home", "pf_home",
    "fgm_away", "fga_away", "fg_pct_away", "fg3m_away", "fg3a_away", "fg3_pct_away",
    "ftm_away", "fta_away", "ft_pct_away", "oreb_away", "dreb_away", "reb_away",
    "ast_away", "stl_away", "blk_away", "tov_away", "pf_away"
]

MODEL_OPTIONS = {
    "Random Forest": "models/random_forest.joblib",
    "Polynomial Logistic": "models/polynomial.joblib",
    "SVM Linear": "models/svm_linear.joblib",
    "SVM RBF": "models/svm_rbf.joblib",
    "SVM Poly": "models/svm_poly.joblib",
    "SVM Sigmoid": "models/svm_sigmoid.joblib",
    "SVM ANOVA": "models/svm_anova.joblib",
    "SVM Cosine": "models/svm_cosine.joblib",
    "SVM Laplacian": "models/svm_laplacian.joblib",
    "SVM Rational Quadratic": "models/svm_rational_quadratic.joblib",
    "GAM": "models/gam.joblib",
    "Spline": "models/spline.joblib",
    "Gradient Boosting": "models/gradientboosting.joblib",
}

NBA_TEAMS = [
    "Atlanta Hawks", "Boston Celtics", "Brooklyn Nets", "Charlotte Hornets",
    "Chicago Bulls", "Cleveland Cavaliers", "Dallas Mavericks", "Denver Nuggets",
    "Detroit Pistons", "Golden State Warriors", "Houston Rockets", "Indiana Pacers",
    "LA Clippers", "Los Angeles Lakers", "Memphis Grizzlies", "Miami Heat",
    "Milwaukee Bucks", "Minnesota Timberwolves", "New Orleans Pelicans", "New York Knicks",
    "Oklahoma City Thunder", "Orlando Magic", "Philadelphia 76ers", "Phoenix Suns",
    "Portland Trail Blazers", "Sacramento Kings", "San Antonio Spurs", "Toronto Raptors",
    "Utah Jazz", "Washington Wizards",
]

TEAM_COLORS = {
    "Atlanta Hawks": "#E03A3E",
    "Boston Celtics": "#007A33",
    "Brooklyn Nets": "#000000",
    "Charlotte Hornets": "#1D1160",
    "Chicago Bulls": "#CE1141",
    "Cleveland Cavaliers": "#6F263D",
    "Dallas Mavericks": "#00538C",
    "Denver Nuggets": "#0E2240",
    "Detroit Pistons": "#C8102E",
    "Golden State Warriors": "#1D428A",
    "Houston Rockets": "#CE1141",
    "Indiana Pacers": "#002D62",
    "LA Clippers": "#C8102E",
    "Los Angeles Lakers": "#552583",
    "Memphis Grizzlies": "#5D76A9",
    "Miami Heat": "#98002E",
    "Milwaukee Bucks": "#00471B",
    "Minnesota Timberwolves": "#0C2340",
    "New Orleans Pelicans": "#0C2340",
    "New York Knicks": "#006BB6",
    "Oklahoma City Thunder": "#007AC1",
    "Orlando Magic": "#0077C0",
    "Philadelphia 76ers": "#006BB6",
    "Phoenix Suns": "#1D1160",
    "Portland Trail Blazers": "#E03A3E",
    "Sacramento Kings": "#5A2D81",
    "San Antonio Spurs": "#C4CED4",
    "Toronto Raptors": "#CE1141",
    "Utah Jazz": "#002B5C",
    "Washington Wizards": "#002B5C",
}

ARENA_THEMES = {
    "Prime Time": {
        "background": "https://images.unsplash.com/photo-1546519638-68e109498ffc?auto=format&fit=crop&w=1800&q=80",
        "overlay_start": "rgba(4, 10, 26, 0.78)",
        "overlay_end": "rgba(8, 22, 52, 0.86)",
    },
    "Hardwood Classic": {
        "background": "https://images.unsplash.com/photo-1518063319789-7217e6706b04?auto=format&fit=crop&w=1800&q=80",
        "overlay_start": "rgba(28, 17, 7, 0.74)",
        "overlay_end": "rgba(60, 34, 12, 0.82)",
    },
    "Playoff Lights": {
        "background": "https://images.unsplash.com/photo-1519861531473-9200262188bf?auto=format&fit=crop&w=1800&q=80",
        "overlay_start": "rgba(14, 11, 4, 0.72)",
        "overlay_end": "rgba(66, 43, 10, 0.84)",
    },
}


def _default_feature_config() -> Dict[str, Dict[str, float]]:
    config: Dict[str, Dict[str, float]] = {}
    for feature in FEATURE_COLUMNS:
        if "pct" in feature:
            config[feature] = {"min": 0.0, "max": 1.0, "default": 0.45, "step": 0.01}
        elif feature.startswith("pts"):
            config[feature] = {"min": 60.0, "max": 180.0, "default": 110.0, "step": 1.0}
        elif feature.startswith("reb"):
            config[feature] = {"min": 20.0, "max": 80.0, "default": 45.0, "step": 1.0}
        elif feature.startswith("fg") or feature.startswith("ft"):
            config[feature] = {"min": 0.0, "max": 80.0, "default": 20.0, "step": 1.0}
        else:
            config[feature] = {"min": 0.0, "max": 40.0, "default": 10.0, "step": 1.0}
    return config


@st.cache_resource
def load_model(model_path: str):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return joblib.load(model_path)


@st.cache_data
def load_data(data_path: str) -> pd.DataFrame:
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    return pd.read_csv(data_path)


def build_feature_config(data: Optional[pd.DataFrame]) -> Dict[str, Dict[str, float]]:
    config = _default_feature_config()
    if data is None:
        return config

    for feature in FEATURE_COLUMNS:
        if feature not in data.columns:
            continue

        series = pd.to_numeric(data[feature], errors="coerce").dropna()
        if series.empty:
            continue

        low = float(series.quantile(0.01))
        high = float(series.quantile(0.99))
        median = float(series.median())

        if np.isclose(low, high):
            continue

        if "pct" in feature:
            low = max(0.0, low)
            high = min(1.0, high)
            step = 0.01
        else:
            low = max(0.0, np.floor(low))
            high = np.ceil(high)
            step = 1.0

        config[feature] = {
            "min": float(low),
            "max": float(high),
            "default": float(np.clip(median, low, high)),
            "step": step,
        }

    return config


def prettify_feature_name(feature: str) -> str:
    side = "Home" if feature.endswith("_home") else "Away"
    base = feature.replace("_home", "").replace("_away", "")
    return f"{side} {base.upper()}"


def build_input_dataframe(inputs: Dict[str, float]) -> pd.DataFrame:
    return pd.DataFrame([[inputs[col] for col in FEATURE_COLUMNS]], columns=FEATURE_COLUMNS)


def team_primary_color(team: str) -> str:
    return TEAM_COLORS.get(team, "#1D428A")


def inject_nba_theme(home_team: str, away_team: str, theme_name: str) -> None:
    selected_theme = ARENA_THEMES.get(theme_name, ARENA_THEMES["Prime Time"])
    home_color = team_primary_color(home_team)
    away_color = team_primary_color(away_team)

    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Barlow:wght@400;500;600;700&display=swap');

        :root {{
            --home-color: {home_color};
            --away-color: {away_color};
        }}

        html, body, [class*="css"] {{
            font-family: 'Barlow', sans-serif;
        }}

        .stApp {{
            background:
                linear-gradient(120deg, {selected_theme["overlay_start"]}, {selected_theme["overlay_end"]}),
                url('{selected_theme["background"]}');
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}

        [data-testid="stHeader"] {{
            background: transparent;
        }}

        .main .block-container {{
            background: rgba(7, 15, 31, 0.70);
            border: 1px solid rgba(255, 255, 255, 0.14);
            border-radius: 20px;
            padding-top: 2.1rem;
            padding-bottom: 1.8rem;
            box-shadow: 0 24px 70px rgba(0, 0, 0, 0.35);
        }}

        [data-testid="stSidebar"] > div:first-child {{
            background: linear-gradient(180deg, rgba(2, 8, 24, 0.98), rgba(9, 24, 52, 0.94));
            border-right: 1px solid rgba(255, 255, 255, 0.12);
        }}

        h1, h2, h3 {{
            font-family: 'Bebas Neue', sans-serif;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            color: #F4F7FF;
        }}

        p, label, .stMarkdown, .stCaption, [data-testid="stMetricLabel"] {{
            color: #EAF0FF !important;
        }}

        [data-testid="stMetricValue"] {{
            color: #FFFFFF;
        }}

        div[data-testid="stMetric"] {{
            background: rgba(255, 255, 255, 0.08);
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-radius: 14px;
            padding: 0.6rem 0.9rem;
        }}

        .stButton > button {{
            background: linear-gradient(132deg, var(--home-color), var(--away-color));
            color: #FFFFFF;
            border: none;
            border-radius: 999px;
            font-weight: 700;
            letter-spacing: 0.03em;
            padding: 0.58rem 1.3rem;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }}

        .stButton > button:hover {{
            transform: translateY(-2px);
            box-shadow: 0 10px 24px rgba(0, 0, 0, 0.35);
        }}

        div[data-baseweb="select"] > div,
        .stTextInput > div > div > input,
        .stNumberInput input,
        textarea,
        input {{
            background-color: rgba(255, 255, 255, 0.08) !important;
            color: #F8FAFF !important;
            border-radius: 10px !important;
            border: 1px solid rgba(255, 255, 255, 0.18) !important;
        }}

        .stSlider [data-baseweb="slider"] > div {{
            background-color: rgba(255, 255, 255, 0.20);
        }}

        [data-testid="stDataFrame"] {{
            background: rgba(255, 255, 255, 0.06);
            border-radius: 14px;
            border: 1px solid rgba(255, 255, 255, 0.16);
        }}

        .hero-board {{
            background: linear-gradient(120deg, rgba(255, 255, 255, 0.12), rgba(255, 255, 255, 0.05));
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-radius: 18px;
            padding: 1rem 1.2rem;
            margin: 0.4rem 0 1.1rem;
            backdrop-filter: blur(4px);
        }}

        .hero-topline {{
            font-size: 0.82rem;
            letter-spacing: 0.12em;
            font-weight: 600;
            text-transform: uppercase;
            color: #D7E3FF;
            margin-bottom: 0.35rem;
        }}

        .hero-title {{
            font-family: 'Bebas Neue', sans-serif;
            font-size: 2.35rem;
            line-height: 1.0;
            letter-spacing: 0.04em;
            color: #FFFFFF;
        }}

        .hero-meta {{
            display: flex;
            flex-wrap: wrap;
            gap: 0.65rem;
            margin-top: 0.7rem;
        }}

        .hero-chip {{
            padding: 0.35rem 0.58rem;
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.12);
            border: 1px solid rgba(255, 255, 255, 0.18);
            font-size: 0.8rem;
            color: #F5F8FF;
            white-space: nowrap;
        }}

        @media (max-width: 900px) {{
            .main .block-container {{
                padding-top: 1.4rem;
            }}

            .hero-title {{
                font-size: 1.8rem;
            }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_matchup_banner(home_team: str, away_team: str, model_name: str, theme_name: str) -> None:
    home_color = team_primary_color(home_team)
    away_color = team_primary_color(away_team)

    st.markdown(
        f"""
        <div class="hero-board">
            <div class="hero-topline">NBA Matchup Intelligence</div>
            <div class="hero-title">{away_team} at {home_team}</div>
            <div class="hero-meta">
                <span class="hero-chip" style="border-left: 5px solid {away_color};">Away: {away_team}</span>
                <span class="hero-chip" style="border-left: 5px solid {home_color};">Home: {home_team}</span>
                <span class="hero-chip">Model: {model_name}</span>
                <span class="hero-chip">Theme: {theme_name}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def predict_outcome(model, input_df: pd.DataFrame) -> Tuple[int, Optional[float]]:
    prediction = int(model.predict(input_df)[0])

    probability: Optional[float] = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(input_df)
        if proba.ndim == 2 and proba.shape[1] >= 2:
            probability = float(proba[0, 1])
    elif hasattr(model, "decision_function"):
        score = float(np.ravel(model.decision_function(input_df))[0])
        probability = float(1.0 / (1.0 + np.exp(-score)))

    return prediction, probability


def render_feature_inputs(config: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    values: Dict[str, float] = {}

    st.sidebar.markdown("### Home Team Stats")
    for feature in [f for f in FEATURE_COLUMNS if f.endswith("_home")]:
        meta = config[feature]
        label = prettify_feature_name(feature)
        if "pct" in feature:
            values[feature] = st.sidebar.slider(
                label,
                min_value=float(meta["min"]),
                max_value=float(meta["max"]),
                value=float(meta["default"]),
                step=float(meta["step"]),
            )
        else:
            values[feature] = float(st.sidebar.number_input(
                label,
                min_value=float(meta["min"]),
                max_value=float(meta["max"]),
                value=float(meta["default"]),
                step=float(meta["step"]),
            ))

    st.sidebar.markdown("### Away Team Stats")
    for feature in [f for f in FEATURE_COLUMNS if f.endswith("_away")]:
        meta = config[feature]
        label = prettify_feature_name(feature)
        if "pct" in feature:
            values[feature] = st.sidebar.slider(
                label,
                min_value=float(meta["min"]),
                max_value=float(meta["max"]),
                value=float(meta["default"]),
                step=float(meta["step"]),
            )
        else:
            values[feature] = float(st.sidebar.number_input(
                label,
                min_value=float(meta["min"]),
                max_value=float(meta["max"]),
                value=float(meta["default"]),
                step=float(meta["step"]),
            ))

    return values


def main() -> None:
    st.set_page_config(page_title="NBA Pro Classification", page_icon=":basketball:", layout="wide")

    st.sidebar.header("Matchup Setup")
    home_team = st.sidebar.selectbox("Home Team", NBA_TEAMS, index=1)
    away_team = st.sidebar.selectbox("Away Team", NBA_TEAMS, index=2)
    theme_name = st.sidebar.selectbox("Arena Theme", list(ARENA_THEMES.keys()), index=0)

    if home_team == away_team:
        st.sidebar.warning("Please select two different teams.")

    model_name = st.sidebar.selectbox("Model", list(MODEL_OPTIONS.keys()), index=0)
    model_path = MODEL_OPTIONS[model_name]

    inject_nba_theme(home_team, away_team, theme_name)

    st.title("NBA Pro Classification")
    st.caption("Professional game-outcome modeling from team box-score signals.")
    render_matchup_banner(home_team, away_team, model_name, theme_name)

    st.sidebar.header("Optional Data Source")
    data_path = st.sidebar.text_input("CSV Path for Feature Ranges", value="data/game.csv")

    data = None
    try:
        data = load_data(data_path)
        st.sidebar.success("Data file loaded.")
    except FileNotFoundError as exc:
        st.sidebar.warning(f"{exc}. Using default input ranges.")
    except Exception as exc:
        st.sidebar.error(f"Could not read data file: {exc}")

    feature_config = build_feature_config(data)
    feature_values = render_feature_inputs(feature_config)

    st.subheader(f"Prediction Desk: {away_team} at {home_team}")

    if st.button("Run Classification", type="primary"):
        try:
            model = load_model(model_path)
        except FileNotFoundError as exc:
            st.error(str(exc))
            st.stop()
        except Exception as exc:
            st.error(f"Failed to load model: {exc}")
            st.stop()

        input_df = build_input_dataframe(feature_values)

        try:
            pred_class, pred_prob = predict_outcome(model, input_df)
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
            st.stop()

        result_label = "Home Win" if pred_class == 1 else "Home Loss"

        if pred_class == 1:
            st.success(f"Classification Result: {result_label} ({home_team} projected to win)")
        else:
            st.warning(f"Classification Result: {result_label} ({away_team} projected to win)")

        if pred_prob is not None:
            home_prob = pred_prob
            away_prob = 1.0 - pred_prob

            metric_col1, metric_col2 = st.columns(2)
            with metric_col1:
                st.metric(f"{home_team} Win Probability", f"{home_prob:.2%}")
            with metric_col2:
                st.metric(f"{away_team} Win Probability", f"{away_prob:.2%}")
        else:
            st.info("This model does not expose probability outputs.")

        st.caption("Model input values used for this prediction")
        st.dataframe(input_df, use_container_width=True)


if __name__ == "__main__":
    main()
