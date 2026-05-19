"""
Phase 4 — Linear Regression Baseline
====================================

Goal: train the simplest reasonable model on model_ready.parquet, evaluate
honestly, and establish the bar that fancier models (Phase 5) must beat.

Steps:
  1. Load model_ready.parquet
  2. Random 80/20 train/test split (safe — 1 sale per parcel = no leakage)
  3. Target-encode BOOK using TRAIN ONLY  -> add BOOK_enc to both sets
  4. One-hot encode PUC_bucketed (categories from train+test combined)
  5. Drop non-feature columns, build X / y
  6. Standardize features (mean=0, std=1)
  7. Train: naive baseline, OLS, Ridge
  8. Evaluate: MAE, RMSE, R² in log space and dollar space
  9. Inspect top coefficients to see what the linear model learned
 10. Save artifacts to models/ for Phase 6 deployment

Why log_price as target? See Phase 2 — SALE_PRICE is right-skewed; log makes
the residuals roughly normal, which is what linear regression assumes.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------------------------------------------------------------------------
# Paths and config
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS = PROJECT_ROOT / "models"
MODELS.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.20
BOOK_SMOOTHING = 50   # pseudo-observations toward the global mean for rare BOOKs
RIDGE_ALPHA = 1.0


# ---------------------------------------------------------------------------
# 1. Load
# ---------------------------------------------------------------------------
def load() -> pd.DataFrame:
    df = pd.read_parquet(PROCESSED / "model_ready.parquet")
    print(f"Loaded {len(df):,} rows × {df.shape[1]} cols")
    return df


# ---------------------------------------------------------------------------
# 2. Split
# ---------------------------------------------------------------------------
def split(df: pd.DataFrame):
    """Random 80/20 split. Safe because each parcel appears exactly once."""
    train, test = train_test_split(df, test_size=TEST_SIZE,
                                   random_state=RANDOM_STATE)
    print(f"  train: {len(train):,}   test: {len(test):,}")
    return train.reset_index(drop=True), test.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 3. Target-encode BOOK (train-only fit)
# ---------------------------------------------------------------------------
def target_encode_book(train: pd.DataFrame, test: pd.DataFrame,
                       smoothing: int = BOOK_SMOOTHING):
    """Replace BOOK with the mean log_price of TRAINING rows in that book.

    Smoothing: rare BOOKs (few sales in train) get pulled toward the global
    mean. The formula adds `smoothing` pseudo-observations at the global mean
    to each book's count before averaging. A BOOK with 1000 training sales
    is barely shrunk; a BOOK with 5 sales is shrunk a lot.

    Returns (train, test, book_map, global_mean) — the latter two are saved
    for Phase 6 prediction.
    """
    global_mean = train["log_price"].mean()
    agg = train.groupby("BOOK")["log_price"].agg(["mean", "count"])
    smoothed = (agg["mean"] * agg["count"] + global_mean * smoothing) \
               / (agg["count"] + smoothing)
    book_map = smoothed.to_dict()

    train["BOOK_enc"] = train["BOOK"].map(book_map)
    # Test BOOKs not seen in train default to the global mean (no info)
    test["BOOK_enc"] = test["BOOK"].map(book_map).fillna(global_mean)

    print(f"  BOOK target-encoded: {len(book_map):,} books, smoothing={smoothing}")
    print(f"    global_mean log_price = {global_mean:.3f}")
    print(f"    encoded range: [{train['BOOK_enc'].min():.3f}, {train['BOOK_enc'].max():.3f}]")
    return train, test, book_map, global_mean


# ---------------------------------------------------------------------------
# 4. One-hot encode PUC_bucketed
# ---------------------------------------------------------------------------
def one_hot_puc(train: pd.DataFrame, test: pd.DataFrame):
    """One-hot encode PUC_bucketed; ensure train and test have identical columns.

    We collect the set of categories from train+test and build dummies
    consistently. Any test-only PUC value gets a column of zeros in train and
    a column of ones in matching test rows.
    """
    all_pucs = sorted(pd.concat([train["PUC_bucketed"],
                                 test["PUC_bucketed"]]).unique())
    for puc in all_pucs:
        col = f"PUC_{puc}"
        train[col] = (train["PUC_bucketed"] == puc).astype(int)
        test[col]  = (test["PUC_bucketed"]  == puc).astype(int)
    train = train.drop(columns=["PUC_bucketed"])
    test  = test.drop(columns=["PUC_bucketed"])
    print(f"  PUC one-hot: +{len(all_pucs)} columns")
    return train, test, all_pucs


# ---------------------------------------------------------------------------
# 5. Build X / y
# ---------------------------------------------------------------------------
DROP_FOR_X = [
    "PARCELNUMBER",   # identifier, not a feature
    "SALE_PRICE",     # un-logged target
    "SALE_DATE",      # we already extracted sale_year / sale_month
    "BOOK",           # replaced by BOOK_enc
    "log_price",      # the target itself
]


def prepare_xy(train: pd.DataFrame, test: pd.DataFrame):
    y_train = train["log_price"].values
    y_test  = test["log_price"].values

    X_train = train.drop(columns=DROP_FOR_X, errors="ignore").copy()
    X_test  = test.drop(columns=DROP_FOR_X,  errors="ignore").copy()

    # Drop any zero-variance columns from training (would break StandardScaler)
    zero_var = X_train.columns[X_train.nunique() <= 1].tolist()
    if zero_var:
        print(f"  dropping {len(zero_var)} zero-variance columns: {zero_var}")
        X_train = X_train.drop(columns=zero_var)
        X_test  = X_test.drop(columns=zero_var)

    print(f"  X_train shape: {X_train.shape}   y_train shape: {y_train.shape}")
    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# 6-7. Standardize + fit
# ---------------------------------------------------------------------------
def evaluate(name: str, y_true_log, y_pred_log) -> dict:
    """Print metrics in log space and dollar space. Returns a dict for logging."""
    y_true_dollar = np.exp(y_true_log)
    y_pred_dollar = np.exp(y_pred_log)

    mae_log  = mean_absolute_error(y_true_log, y_pred_log)
    rmse_log = mean_squared_error(y_true_log, y_pred_log) ** 0.5
    r2_log   = r2_score(y_true_log, y_pred_log)

    mae_d  = mean_absolute_error(y_true_dollar, y_pred_dollar)
    rmse_d = mean_squared_error(y_true_dollar, y_pred_dollar) ** 0.5
    r2_d   = r2_score(y_true_dollar, y_pred_dollar)

    # MAPE — Mean Absolute Percent Error. Tells you "off by ~N%" intuitively.
    mape = np.mean(np.abs((y_pred_dollar - y_true_dollar) / y_true_dollar)) * 100

    print(f"\n  {name}")
    print(f"    log  MAE: {mae_log:.4f}   RMSE: {rmse_log:.4f}   R²: {r2_log:+.4f}")
    print(f"    $    MAE: ${mae_d:>9,.0f}   RMSE: ${rmse_d:>9,.0f}   R²: {r2_d:+.4f}   MAPE: {mape:.2f}%")
    return dict(name=name, mae_log=mae_log, rmse_log=rmse_log, r2_log=r2_log,
                mae_d=mae_d, rmse_d=rmse_d, r2_d=r2_d, mape=mape)


def train_models(X_train, X_test, y_train, y_test):
    """Scale, fit two models, plus a naive baseline. Returns models + scaler."""
    scaler = StandardScaler()
    Xt = scaler.fit_transform(X_train)
    Xv = scaler.transform(X_test)

    print("\n=== EVALUATION ON TEST SET ===")

    # Naive baseline: predict the mean training log_price for every row
    naive_pred = np.full(shape=y_test.shape, fill_value=y_train.mean())
    evaluate("Baseline (predict mean)", y_test, naive_pred)

    ols = LinearRegression()
    ols.fit(Xt, y_train)
    evaluate("OLS", y_test, ols.predict(Xv))

    ridge = Ridge(alpha=RIDGE_ALPHA, random_state=RANDOM_STATE)
    ridge.fit(Xt, y_train)
    evaluate(f"Ridge (alpha={RIDGE_ALPHA})", y_test, ridge.predict(Xv))

    return ols, ridge, scaler


# ---------------------------------------------------------------------------
# 8. Inspect top coefficients
# ---------------------------------------------------------------------------
def show_top_coefficients(model, feature_names, n: int = 15):
    """Print the n features with the largest |coefficient| in the scaled model."""
    coefs = pd.Series(model.coef_, index=feature_names)
    ranked = coefs.reindex(coefs.abs().sort_values(ascending=False).index).head(n)
    print(f"\n=== TOP {n} FEATURES BY |COEFFICIENT| ===")
    print(ranked.to_string(float_format=lambda x: f"{x:+.4f}"))
    print("  (positive = pushes price up,  negative = pushes price down)")
    print("  (coefficients are on standardized features — directly comparable)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    df = load()

    train, test = split(df)
    train, test, book_map, global_mean = target_encode_book(train, test)
    train, test, puc_categories = one_hot_puc(train, test)
    X_train, X_test, y_train, y_test = prepare_xy(train, test)

    ols, ridge, scaler = train_models(X_train, X_test, y_train, y_test)
    show_top_coefficients(ridge, X_train.columns)

    # Save artifacts for Phase 6 (Streamlit prediction)
    joblib.dump(ols,   MODELS / "ols.joblib")
    joblib.dump(ridge, MODELS / "ridge.joblib")
    joblib.dump(scaler, MODELS / "scaler.joblib")
    joblib.dump(
        {
            "feature_names": list(X_train.columns),
            "book_map": book_map,
            "global_mean": global_mean,
            "puc_categories": puc_categories,
        },
        MODELS / "preprocess.joblib",
    )
    print(f"\nSaved 4 artifacts to {MODELS}/")
    print("  ols.joblib, ridge.joblib, scaler.joblib, preprocess.joblib")


if __name__ == "__main__":
    main()
