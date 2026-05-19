"""
Phase 6 — Streamlit Deployment
==============================

A small web app that loads our trained Random Forest model and serves
price predictions for user-submitted property characteristics.

Run from the project root:
    streamlit run app.py

The app:
  1. Loads model + scaler + preprocess metadata (cached so it happens once)
  2. Presents ~15 form fields for property characteristics
  3. On submit:
       a. Builds a 79-column feature row from the inputs
       b. Applies BOOK target encoding, PUC one-hot, derived features
       c. Reindexes to match training column order
       d. Standardizes with the saved scaler
       e. Predicts log_price; exponentiates to dollars
       f. Computes an 80% interval from per-tree spread (RF) or RMSE (XGB)
       g. Shows the top features driving the prediction
"""

from pathlib import Path
from datetime import datetime
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Page config — must be the first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Maricopa Home Price Predictor",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
MODELS = HERE / "models"


# ---------------------------------------------------------------------------
# Load artifacts (cached so this runs once across all reruns)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    pp = joblib.load(MODELS / "preprocess.joblib")
    scaler = joblib.load(MODELS / "scaler.joblib")
    winner = pp.get("winner", "Random Forest")
    model_file = "random_forest.joblib" if "Random Forest" in winner else "xgboost.joblib"
    model = joblib.load(MODELS / model_file)
    return model, scaler, pp, winner


model, scaler, pp, winner_name = load_artifacts()
FEATURE_NAMES = pp["feature_names"]
BOOK_MAP = pp["book_map"]
GLOBAL_MEAN = pp["global_mean"]
PUC_CATEGORIES = pp["puc_categories"]

# ---------------------------------------------------------------------------
# Helpers — extract category options directly from the one-hot column names,
# so the app stays in sync with whatever the training data actually had.
# ---------------------------------------------------------------------------
def options_for(prefix: str) -> list[str]:
    return sorted(n[len(prefix):] for n in FEATURE_NAMES if n.startswith(prefix))


AC_OPTS    = options_for("AirConditioningType_")
HEAT_OPTS  = options_for("HeatingType_")
WALL_OPTS  = options_for("ExteriorWallMaterial_")
ROOF_OPTS  = options_for("RoofMaterial_")
BOOK_OPTS  = sorted(BOOK_MAP.keys())
PUC_OPTS   = sorted(PUC_CATEGORIES)

CLASS_MAP = {
    "LOW MINIMUM": 0, "CLASS R1": 1, "CLASS R2": 2, "CLASS R3": 3,
    "CLASS R4": 4,    "CLASS R5": 5, "CLASS R6": 6, "CLASS R7": 7,
}


def safe_index(options: list, target: str, fallback: int = 0) -> int:
    return options.index(target) if target in options else fallback


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("Maricopa County Home Price Predictor")
st.caption(
    f"Powered by {winner_name} — trained on 365,076 Maricopa County sales (2018-2026)"
)

# ---------------------------------------------------------------------------
# Input form
# ---------------------------------------------------------------------------
with st.form("predict_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Basics")
        living_sqft = st.number_input(
            "Living square feet", min_value=300, max_value=10_000,
            value=2_000, step=100,
        )
        construction_year = st.number_input(
            "Year built", min_value=1900, max_value=2026, value=2000,
        )
        story_count = st.selectbox("Stories", [1, 2, 3], index=0)
        bathroom_fixtures = st.number_input(
            "Bathroom fixtures", min_value=1, max_value=20, value=6,
            help="Total count of fixtures (toilets, sinks, tubs/showers).",
        )

    with col2:
        st.subheader("Location & Type")
        book = st.selectbox(
            "BOOK (geographic cluster from APN)",
            BOOK_OPTS,
            index=safe_index(BOOK_OPTS, "050"),
            help="First 3 digits of the parcel number. BOOKs cluster geographically.",
        )
        puc = st.selectbox(
            "Property Use Code (PUC)",
            PUC_OPTS,
            index=safe_index(PUC_OPTS, "131"),
        )
        class_choice = st.selectbox(
            "Property Class",
            list(CLASS_MAP.keys()),
            index=3,  # default CLASS R3 (the bulk of the market)
        )

    with col3:
        st.subheader("Materials")
        ac = st.selectbox(
            "Cooling", AC_OPTS,
            index=safe_index(AC_OPTS, "RF - REFRIGERATION"),
        )
        heating = st.selectbox(
            "Heating", HEAT_OPTS,
            index=safe_index(HEAT_OPTS, "Yes"),
        )
        wall = st.selectbox(
            "Wall material", WALL_OPTS,
            index=safe_index(WALL_OPTS, "FS - FRAME STUCCO"),
        )
        roof = st.selectbox(
            "Roof material", ROOF_OPTS,
            index=safe_index(ROOF_OPTS, "CT - CONCRETE TILE"),
        )

    st.subheader("Features")
    fcol1, fcol2, fcol3, fcol4 = st.columns(4)

    with fcol1:
        has_pool = st.checkbox("Has pool")
        pool_sqft = st.number_input("Pool sqft", 0, 2000, 400,
                                    disabled=not has_pool)

    with fcol2:
        has_garage = st.checkbox("Has garage", value=True)
        garage_spaces = st.number_input("Garage spaces", 0, 5, 2,
                                        disabled=not has_garage)
        garage_sqft = st.number_input("Garage sqft", 0, 2000, 400,
                                      disabled=not has_garage)

    with fcol3:
        has_patio = st.checkbox("Has patio", value=True)
        covered_patio = st.number_input("Covered patio sqft", 0, 2000, 200,
                                        disabled=not has_patio)
        uncovered_patio = st.number_input("Uncovered patio sqft", 0, 2000, 0,
                                          disabled=not has_patio)

    with fcol4:
        has_basement = st.checkbox("Has basement")
        basement_sqft = st.number_input("Basement sqft", 0, 5000, 0,
                                        disabled=not has_basement)

    submitted = st.form_submit_button(
        "Predict price", type="primary", use_container_width=True
    )


# ---------------------------------------------------------------------------
# Build feature row + predict
# ---------------------------------------------------------------------------
def build_feature_row(inputs: dict) -> pd.DataFrame:
    """Construct the 79-column feature vector the model expects, in order."""
    today = datetime.now()
    age = max(0, today.year - inputs["construction_year"])

    # Distribute living area across floors. For 1-story, everything is on floor 1.
    # For multi-story, put a slightly larger share on floor 1, remainder on floor 2.
    if inputs["story_count"] == 1:
        first_floor = inputs["living_sqft"]
        second_floor = 0
    else:
        first_floor = (inputs["living_sqft"] + 1) // inputs["story_count"]
        second_floor = inputs["living_sqft"] - first_floor

    raw = {
        # Numerics
        "StoryCount":        inputs["story_count"],
        "BathroomFixtures":  inputs["bathroom_fixtures"],
        "Living_sqft":       inputs["living_sqft"],
        "FirstFloor_sqft":   first_floor,
        "SecondFloor_sqft":  second_floor,
        "ThirdFloor_sqft":   0,
        "Basement_sqft":     inputs["basement_sqft"] if inputs["has_basement"] else 0,
        "ConstructionYear":  inputs["construction_year"],
        "POOL_SQFT":         inputs["pool_sqft"] if inputs["has_pool"] else 0,
        "ADDED_SQFT":        0,
        "DETACH_SQFT":       0,
        "sale_year":         today.year,
        "sale_month":        today.month,
        "age_at_sale":       age,
        "has_pool":          int(inputs["has_pool"]),
        "has_basement":      int(inputs["has_basement"]),

        # Parking
        "garage_sqft":       inputs["garage_sqft"]   if inputs["has_garage"] else 0,
        "garage_spaces":     inputs["garage_spaces"] if inputs["has_garage"] else 0,
        "carport_sqft":      0,
        "carport_spaces":    0,
        "rv_sqft":           0,
        "rv_spaces":         0,
        "has_garage":        int(inputs["has_garage"]),

        # Patios
        "covered_patio_sqft":   inputs["covered_patio"]   if inputs["has_patio"] else 0,
        "uncovered_patio_sqft": inputs["uncovered_patio"] if inputs["has_patio"] else 0,
        "has_patio":            int(inputs["has_patio"]),

        # Encoded categoricals
        "class_ord":         CLASS_MAP[inputs["class_choice"]],
        "BOOK_enc":          BOOK_MAP.get(inputs["book"], GLOBAL_MEAN),
    }

    # One-hot AC / Heating / Walls / Roof / PUC
    for opt in AC_OPTS:
        raw[f"AirConditioningType_{opt}"] = int(opt == inputs["ac"])
    for opt in HEAT_OPTS:
        raw[f"HeatingType_{opt}"] = int(opt == inputs["heating"])
    for opt in WALL_OPTS:
        raw[f"ExteriorWallMaterial_{opt}"] = int(opt == inputs["wall"])
    for opt in ROOF_OPTS:
        raw[f"RoofMaterial_{opt}"] = int(opt == inputs["roof"])
    for c in PUC_CATEGORIES:
        raw[f"PUC_{c}"] = int(c == inputs["puc"])

    # Reindex to match training column order. Missing columns get 0.
    row = pd.DataFrame([raw]).reindex(columns=FEATURE_NAMES, fill_value=0)
    return row


def predict(row_df: pd.DataFrame) -> tuple[float, float, float]:
    """Return (log_mean, dollar_mean, log_std). 80% CI = ±1.28 σ."""
    row_scaled = scaler.transform(row_df)

    if hasattr(model, "estimators_"):
        # Random Forest — get a prediction from each of the 200 trees
        tree_preds = np.array([tree.predict(row_scaled)[0]
                               for tree in model.estimators_])
        log_mean = tree_preds.mean()
        log_std = tree_preds.std()
    else:
        # XGBoost — no per-tree predictions easily available; use overall RMSE
        log_mean = float(model.predict(row_scaled)[0])
        log_std = 0.225  # ≈ test-set RMSE in log space

    return log_mean, float(np.exp(log_mean)), log_std


def feature_contributions(row_df: pd.DataFrame, n: int = 8) -> pd.DataFrame:
    """For the top-n features by importance, show user value vs training avg."""
    importance = pd.Series(model.feature_importances_, index=FEATURE_NAMES)
    top = importance.sort_values(ascending=False).head(n).index.tolist()

    rows = []
    for f in top:
        idx = FEATURE_NAMES.index(f)
        user_val   = row_df[f].iloc[0]
        train_mean = scaler.mean_[idx]
        train_std  = scaler.scale_[idx]
        if train_std > 1e-9:
            z = (user_val - train_mean) / train_std
            if   z >  0.30: direction = "↑ above avg"
            elif z < -0.30: direction = "↓ below avg"
            else:           direction = "≈ average"
        else:
            direction = "—"
        rows.append({
            "feature":      f,
            "your value":   f"{user_val:,.1f}".rstrip("0").rstrip("."),
            "training avg": f"{train_mean:,.1f}",
            "vs training":  direction,
            "importance":   f"{importance[f]:.3f}",
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# On submit
# ---------------------------------------------------------------------------
if submitted:
    inputs = dict(
        living_sqft=living_sqft, construction_year=construction_year,
        story_count=story_count, bathroom_fixtures=bathroom_fixtures,
        book=book, puc=puc, class_choice=class_choice,
        ac=ac, heating=heating, wall=wall, roof=roof,
        has_pool=has_pool, pool_sqft=pool_sqft,
        has_garage=has_garage, garage_spaces=garage_spaces, garage_sqft=garage_sqft,
        has_patio=has_patio, covered_patio=covered_patio, uncovered_patio=uncovered_patio,
        has_basement=has_basement, basement_sqft=basement_sqft,
    )

    row = build_feature_row(inputs)
    log_mean, dollar_mean, log_std = predict(row)
    dollar_lo = float(np.exp(log_mean - 1.28 * log_std))
    dollar_hi = float(np.exp(log_mean + 1.28 * log_std))

    # ---- Display ----
    st.divider()
    st.subheader("Prediction")

    m1, m2, m3 = st.columns(3)
    m1.metric("Estimated price",      f"${dollar_mean:,.0f}")
    m2.metric("80% interval — low",   f"${dollar_lo:,.0f}")
    m3.metric("80% interval — high",  f"${dollar_hi:,.0f}")

    st.subheader("What's driving this prediction?")
    st.dataframe(
        feature_contributions(row),
        hide_index=True, use_container_width=True,
    )

    st.caption(
        f"Model: {winner_name}. Predictions reflect 2018-2026 training data. "
        f"The 80% interval comes from how much the 200 trees disagree on this input — "
        f"wider = more uncertainty."
    )
