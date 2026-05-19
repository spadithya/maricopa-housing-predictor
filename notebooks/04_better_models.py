"""
Phase 5 — Random Forest and Gradient Boosting
==============================================

Goal: beat the Phase 4 linear baseline using ensemble tree models.
Same train/test split, same target encoding, same scaler (for fairness), but
swap the estimator for tree-based learners that capture nonlinearities and
interactions.

Steps:
  1. Reproduce the Phase 4 preprocessing exactly (load, split, BOOK encode,
     PUC one-hot, drop non-features, standardize)
  2. Train RandomForestRegressor
  3. Train XGBoost
  4. Evaluate both with the same metrics as Phase 4 — direct comparison
  5. Inspect feature importances (different concept than coefficients!)
  6. Save the better of the two as the production model

Note on scaling: tree models don't NEED scaling (they split on thresholds,
which are scale-invariant), but we standardize anyway for consistency with
the linear baseline. The trees produce identical results either way.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from xgboost import XGBRegressor
    HAVE_XGB = True
except ImportError:
    HAVE_XGB = False
    print("WARNING: xgboost not installed. Run `pip install xgboost`.")

# ---------------------------------------------------------------------------
# Paths and config — match Phase 4 exactly
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS = PROJECT_ROOT / "models"
MODELS.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.20
BOOK_SMOOTHING = 50


# ---------------------------------------------------------------------------
# Preprocessing — copied verbatim from Phase 4 so models train on the SAME features
# ---------------------------------------------------------------------------
def load() -> pd.DataFrame:
    df = pd.read_parquet(PROCESSED / "model_ready.parquet")
    print(f"Loaded {len(df):,} rows × {df.shape[1]} cols")
    return df


def split(df):
    train, test = train_test_split(df, test_size=TEST_SIZE,
                                   random_state=RANDOM_STATE)
    return train.reset_index(drop=True), test.reset_index(drop=True)


def target_encode_book(train, test, smoothing=BOOK_SMOOTHING):
    global_mean = train["log_price"].mean()
    agg = train.groupby("BOOK")["log_price"].agg(["mean", "count"])
    smoothed = (agg["mean"] * agg["count"] + global_mean * smoothing) \
               / (agg["count"] + smoothing)
    book_map = smoothed.to_dict()
    train["BOOK_enc"] = train["BOOK"].map(book_map)
    test["BOOK_enc"]  = test["BOOK"].map(book_map).fillna(global_mean)
    return train, test, book_map, global_mean


def one_hot_puc(train, test):
    all_pucs = sorted(pd.concat([train["PUC_bucketed"],
                                 test["PUC_bucketed"]]).unique())
    for puc in all_pucs:
        col = f"PUC_{puc}"
        train[col] = (train["PUC_bucketed"] == puc).astype(int)
        test[col]  = (test["PUC_bucketed"]  == puc).astype(int)
    train = train.drop(columns=["PUC_bucketed"])
    test  = test.drop(columns=["PUC_bucketed"])
    return train, test, all_pucs


DROP_FOR_X = ["PARCELNUMBER", "SALE_PRICE", "SALE_DATE", "BOOK", "log_price"]


def prepare_xy(train, test):
    y_train = train["log_price"].values
    y_test  = test["log_price"].values
    X_train = train.drop(columns=DROP_FOR_X, errors="ignore").copy()
    X_test  = test.drop(columns=DROP_FOR_X,  errors="ignore").copy()
    zero_var = X_train.columns[X_train.nunique() <= 1].tolist()
    if zero_var:
        X_train = X_train.drop(columns=zero_var)
        X_test  = X_test.drop(columns=zero_var)
    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# Evaluation — same as Phase 4
# ---------------------------------------------------------------------------
def evaluate(name, y_true_log, y_pred_log):
    y_true_d = np.exp(y_true_log)
    y_pred_d = np.exp(y_pred_log)
    mae_log  = mean_absolute_error(y_true_log, y_pred_log)
    rmse_log = mean_squared_error(y_true_log, y_pred_log) ** 0.5
    r2_log   = r2_score(y_true_log, y_pred_log)
    mae_d    = mean_absolute_error(y_true_d, y_pred_d)
    rmse_d   = mean_squared_error(y_true_d, y_pred_d) ** 0.5
    r2_d     = r2_score(y_true_d, y_pred_d)
    mape     = np.mean(np.abs((y_pred_d - y_true_d) / y_true_d)) * 100
    print(f"\n  {name}")
    print(f"    log  MAE: {mae_log:.4f}   RMSE: {rmse_log:.4f}   R²: {r2_log:+.4f}")
    print(f"    $    MAE: ${mae_d:>9,.0f}   RMSE: ${rmse_d:>9,.0f}   "
          f"R²: {r2_d:+.4f}   MAPE: {mape:.2f}%")
    return dict(name=name, mae_log=mae_log, rmse_log=rmse_log, r2_log=r2_log,
                mae_d=mae_d, rmse_d=rmse_d, r2_d=r2_d, mape=mape)


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------
def train_random_forest(Xt, y_train, Xv, y_test):
    """Random Forest — an average of many decorrelated decision trees.

    Hyperparams:
      n_estimators=200   — number of trees (more = better but slower)
      max_depth=None     — let each tree grow until pure (RF is robust to overfitting)
      min_samples_leaf=5 — don't split if a leaf would have <5 rows (smooth predictions)
      n_jobs=-1          — use all CPU cores
    """
    print("\nTraining Random Forest (this can take 1-3 minutes)...")
    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    rf.fit(Xt, y_train)
    metrics = evaluate("Random Forest", y_test, rf.predict(Xv))
    return rf, metrics


def train_xgboost(Xt, y_train, Xv, y_test):
    """XGBoost — gradient-boosted trees, the workhorse of tabular ML.

    Hyperparams:
      n_estimators=500     — boosting rounds (more rounds + lower LR = better)
      learning_rate=0.05   — shrink each tree's contribution (regularization)
      max_depth=6          — moderate tree depth
      subsample=0.8        — each tree sees 80% of rows (stochastic boosting)
      colsample_bytree=0.8 — each tree sees 80% of features
      reg_lambda=1.0       — L2 on leaf weights
      n_jobs=-1            — all cores
    """
    if not HAVE_XGB:
        return None, None
    print("\nTraining XGBoost (this should take under a minute)...")
    xgb = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=0,
    )
    xgb.fit(Xt, y_train)
    metrics = evaluate("XGBoost", y_test, xgb.predict(Xv))
    return xgb, metrics


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------
def show_importances(model, feature_names, name: str, n: int = 15):
    """Print top-n features by built-in importance score.

    For RF and XGBoost, `feature_importances_` is "how much does using this
    feature reduce impurity across all splits, on average?" — a different
    concept from a linear model's coefficient.
    """
    imp = pd.Series(model.feature_importances_, index=feature_names) \
            .sort_values(ascending=False)
    print(f"\n=== TOP {n} FEATURE IMPORTANCES ({name}) ===")
    print(imp.head(n).to_string(float_format=lambda x: f"{x:.4f}"))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    df = load()
    train, test = split(df)
    train, test, book_map, global_mean = target_encode_book(train, test)
    train, test, puc_categories = one_hot_puc(train, test)
    X_train, X_test, y_train, y_test = prepare_xy(train, test)

    print(f"  X_train shape: {X_train.shape}   y_train shape: {y_train.shape}")

    # Scale once — trees don't need it but matches Phase 4's setup
    scaler = StandardScaler()
    Xt = scaler.fit_transform(X_train)
    Xv = scaler.transform(X_test)

    print("\n=== EVALUATION ON TEST SET ===")

    rf, rf_metrics = train_random_forest(Xt, y_train, Xv, y_test)
    xgb, xgb_metrics = train_xgboost(Xt, y_train, Xv, y_test)

    show_importances(rf, X_train.columns, "Random Forest")
    if xgb is not None:
        show_importances(xgb, X_train.columns, "XGBoost")

    # Pick the winner by log-space MAE (the model's actual loss)
    candidates = [m for m in [rf_metrics, xgb_metrics] if m is not None]
    winner = min(candidates, key=lambda m: m["mae_log"])
    print(f"\n=== WINNER: {winner['name']} (log MAE = {winner['mae_log']:.4f}) ===")

    # Save artifacts
    joblib.dump(rf, MODELS / "random_forest.joblib")
    if xgb is not None:
        joblib.dump(xgb, MODELS / "xgboost.joblib")

    # Overwrite scaler + preprocess in case downstream Phase 6 wants the
    # Phase 5 preprocessing instead of Phase 4's (they're identical, but
    # being explicit avoids stale-model bugs)
    joblib.dump(scaler, MODELS / "scaler.joblib")
    joblib.dump(
        {
            "feature_names": list(X_train.columns),
            "book_map": book_map,
            "global_mean": global_mean,
            "puc_categories": puc_categories,
            "winner": winner["name"],
        },
        MODELS / "preprocess.joblib",
    )
    print(f"\nSaved models + preprocessing to {MODELS}/")


if __name__ == "__main__":
    main()
