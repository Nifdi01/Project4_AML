import os
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import streamlit as st


# Custom kernel functions
def kernel_laplacian(X, Y):
    """Laplacian (L1-RBF) — more robust to outliers than Gaussian RBF"""
    from sklearn.metrics.pairwise import manhattan_distances
    gamma = 1.0 / X.shape[1]
    return np.exp(-gamma * manhattan_distances(X, Y))

def kernel_rational_quadratic(X, Y):
    """Rational Quadratic — mixture of RBFs at different scales"""
    alpha, c = 1.0, 1.0
    from sklearn.metrics.pairwise import euclidean_distances
    D2 = euclidean_distances(X, Y) ** 2
    return (1 - D2 / (D2 + c)) ** alpha

def kernel_anova(X, Y, sigma=1.0, d=2):
    """ANOVA kernel — captures feature interaction effects"""
    K = np.zeros((X.shape[0], Y.shape[0]))
    for k in range(X.shape[1]):
        K += np.exp(-sigma * (X[:, k:k+1] - Y[:, k].reshape(1, -1)) ** 2) ** d
    return K

def kernel_cosine_custom(X, Y):
    """Cosine similarity — useful when direction matters more than magnitude"""
    from sklearn.metrics.pairwise import cosine_similarity
    return cosine_similarity(X, Y)


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
        "background": "https://c4.wallpaperflare.com/wallpaper/441/390/498/basketball-lebron-james-nba-wallpaper-preview.jpg",
        "overlay_start": "rgba(4, 10, 26, 0.86)",
        "overlay_end": "rgba(8, 22, 52, 0.93)",
    },
    "Hardwood Classic": {
        "background": "https://c4.wallpaperflare.com/wallpaper/441/390/498/basketball-lebron-james-nba-wallpaper-preview.jpg",
        "overlay_start": "rgba(28, 17, 7, 0.84)",
        "overlay_end": "rgba(60, 34, 12, 0.91)",
    },
    "Playoff Lights": {
        "background": "https://c4.wallpaperflare.com/wallpaper/441/390/498/basketball-lebron-james-nba-wallpaper-preview.jpg",
        "overlay_start": "rgba(14, 11, 4, 0.84)",
        "overlay_end": "rgba(66, 43, 10, 0.92)",
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


def call_model_method_with_fallback(model: Any, method_name: str, input_df: pd.DataFrame):
    method = getattr(model, method_name, None)
    if method is None:
        return None

    try:
        return method(input_df)
    except Exception as first_exc:
        # Some serialized models use NumPy-style indexing (X[:, i:j]) and fail on DataFrames.
        try:
            return method(input_df.to_numpy(dtype=float))
        except Exception:
            raise first_exc


def _normalize_label(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def infer_home_win_class(model: Any) -> Optional[Any]:
    classes = getattr(model, "classes_", None)
    if classes is None:
        return None

    class_list = [_normalize_label(cls) for cls in np.ravel(classes)]

    for cls in class_list:
        if isinstance(cls, str):
            label = cls.strip().lower()
            if "home" in label and "win" in label:
                return cls

    for cls in class_list:
        if isinstance(cls, (bool, np.bool_)) and bool(cls):
            return cls

    for cls in class_list:
        if isinstance(cls, (int, float, np.integer, np.floating)) and np.isclose(float(cls), 1.0):
            return cls

    return None


def is_home_win_prediction(model: Any, raw_prediction: Any) -> bool:
    prediction = _normalize_label(raw_prediction)
    home_win_class = infer_home_win_class(model)

    if home_win_class is not None:
        return prediction == home_win_class

    if isinstance(prediction, str):
        label = prediction.strip().lower()
        if "away" in label and "win" in label:
            return False
        if "home" in label and "win" in label:
            return True
        if "loss" in label:
            return False
        if "win" in label:
            return True

    if isinstance(prediction, (bool, np.bool_)):
        return bool(prediction)

    if isinstance(prediction, (int, float, np.integer, np.floating)):
        return np.isclose(float(prediction), 1.0)

    return bool(prediction)


def extract_home_win_probability(model: Any, proba: np.ndarray) -> Optional[float]:
    if proba.ndim != 2 or proba.shape[1] < 2:
        return None

    classes = getattr(model, "classes_", None)
    if classes is not None:
        class_list = [_normalize_label(cls) for cls in np.ravel(classes)]
        if len(class_list) == proba.shape[1]:
            home_win_class = infer_home_win_class(model)
            if home_win_class is not None and home_win_class in class_list:
                home_idx = class_list.index(home_win_class)
                return float(proba[0, home_idx])

            if 1 in class_list:
                return float(proba[0, class_list.index(1)])

            if True in class_list:
                return float(proba[0, class_list.index(True)])

    return float(proba[0, 1])


def extract_home_win_probability_from_decision(model: Any, score: float, is_home_win: bool) -> float:
    positive_class_probability = float(1.0 / (1.0 + np.exp(-score)))

    classes = getattr(model, "classes_", None)
    if classes is not None:
        class_list = [_normalize_label(cls) for cls in np.ravel(classes)]
        if len(class_list) == 2:
            home_win_class = infer_home_win_class(model)
            if home_win_class is not None and home_win_class in class_list:
                return positive_class_probability if class_list[1] == home_win_class else float(1.0 - positive_class_probability)

    # Fallback: force consistency between class label and displayed percentages.
    if is_home_win and positive_class_probability < 0.5:
        return float(1.0 - positive_class_probability)
    if (not is_home_win) and positive_class_probability > 0.5:
        return float(1.0 - positive_class_probability)
    return positive_class_probability


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
            background: rgba(7, 15, 31, 0.90);
            border: 1px solid rgba(255, 255, 255, 0.22);
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
            background: #11284f;
            border: 1px solid #355c97;
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
            background-color: rgba(15, 37, 71, 0.90) !important;
            color: #F8FAFF !important;
            border-radius: 10px !important;
            border: 1px solid rgba(255, 255, 255, 0.30) !important;
        }}

        .stSlider [data-baseweb="slider"] > div {{
            background-color: rgba(255, 255, 255, 0.36);
        }}

        [data-testid="stDataFrame"] {{
            background: rgba(14, 30, 60, 0.88);
            border-radius: 14px;
            border: 1px solid rgba(255, 255, 255, 0.25);
        }}

        .hero-board {{
            background: #0e2346;
            border: 1px solid #355c97;
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
            background: rgba(34, 65, 114, 0.95);
            border: 1px solid rgba(255, 255, 255, 0.28);
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


def predict_outcome(model, input_df: pd.DataFrame) -> Tuple[bool, Optional[float]]:
    pred_values = call_model_method_with_fallback(model, "predict", input_df)
    if pred_values is None:
        raise AttributeError("Model does not implement predict().")

    prediction = np.ravel(pred_values)[0]
    is_home_win = is_home_win_prediction(model, prediction)

    probability: Optional[float] = None
    proba = call_model_method_with_fallback(model, "predict_proba", input_df)
    if proba is not None:
        probability = extract_home_win_probability(model, np.asarray(proba))
    else:
        decision = call_model_method_with_fallback(model, "decision_function", input_df)
        if decision is None:
            return is_home_win, probability

        score = float(np.ravel(decision)[0])
        probability = extract_home_win_probability_from_decision(model, score, is_home_win)

    if probability is not None:
        if is_home_win and probability < 0.5:
            probability = float(1.0 - probability)
        elif (not is_home_win) and probability > 0.5:
            probability = float(1.0 - probability)

    return is_home_win, probability


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
    theme_name = "Prime Time"
    st.sidebar.caption("Theme: Prime Time")

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
            is_home_win, pred_prob = predict_outcome(model, input_df)
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
            st.stop()


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
