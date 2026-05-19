"""
Phase 3 — Cleaning and Feature Engineering
==========================================

Input:  data/processed/residential_with_sales.parquet  (456k clean sales)
Output: data/processed/model_ready.parquet             (same rows, model-ready columns)

What we do, in order:
  1. Drop dead columns (RoofStyle 100% null, ProportionComplete zero-variance)
  2. Outlier handling on Living_sqft (drop tiny, clip huge)
  3. Engineer numerics: sale_year/month, age_at_sale, has_pool, has_basement,
     log_price
  4. Parse PARK_CODE  "G2-380;R1-645"  -> garage/carport/rv counts and sqft
  5. Parse Patios     "CV-500;UC-700"  -> covered/uncovered patio sqft
  6. Encode Class as ordinal (it's a tiered quality code)
  7. One-hot encode small categoricals (AC, Heating, ExteriorWall, RoofMaterial)
  8. Bucket PUC: top 20 by count, lump the rest as 'OTHER'
  9. Keep BOOK as a categorical string (target encoding happens in Phase 4
     with proper train-only fitting)

Design choice: feature parsing lives in plain functions so we can reuse them
in Phase 6 (Streamlit prediction) on user-supplied inputs.
"""

from pathlib import Path
import re
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
PROCESSED = PROJECT_ROOT / "data" / "processed"


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
def load() -> pd.DataFrame:
    df = pd.read_parquet(PROCESSED / "residential_with_sales.parquet")
    print(f"Loaded {len(df):,} rows, {df.shape[1]} cols")
    return df


# ---------------------------------------------------------------------------
# 1 & 2. Drops and outliers
# ---------------------------------------------------------------------------
def drop_and_clip(df: pd.DataFrame) -> pd.DataFrame:
    """Drop dead columns; clip Living_sqft outliers."""
    df = df.copy()
    n0 = len(df)

    # Dead columns — RoofStyle (100% NaN) and ProportionComplete (zero variance)
    df = df.drop(columns=["RoofStyle", "ProportionComplete"], errors="ignore")

    # Tiny homes are data errors
    df = df[df["Living_sqft"] >= 300].copy()

    # Cap huge homes (real mansions but unrepresentative)
    df["Living_sqft"] = df["Living_sqft"].clip(upper=10_000)

    print(f"  drop+clip: {n0:,} -> {len(df):,} rows")
    return df


# ---------------------------------------------------------------------------
# 3. Numeric feature engineering
# ---------------------------------------------------------------------------
def engineer_numerics(df: pd.DataFrame) -> pd.DataFrame:
    """Add sale_year/month, age_at_sale, has_pool, has_basement, log_price."""
    df = df.copy()

    df["sale_year"]  = df["SALE_DATE"].dt.year
    df["sale_month"] = df["SALE_DATE"].dt.month

    df["age_at_sale"] = df["sale_year"] - df["ConstructionYear"]
    # Clip negative ages (pre-construction sales — rare data quirks)
    df["age_at_sale"] = df["age_at_sale"].clip(lower=0)

    df["has_pool"]     = (df["POOL_SQFT"]     > 0).astype(int)
    df["has_basement"] = (df["Basement_sqft"] > 0).astype(int)

    # The modeling target — log because price is log-normal
    df["log_price"] = np.log(df["SALE_PRICE"])

    print(f"  numerics engineered: +{6} columns")
    return df


# ---------------------------------------------------------------------------
# 4. Parse PARK_CODE
# ---------------------------------------------------------------------------
# Format from FileSpec:
#   "<Type><Vehicles>-<SqFt>; <Type><Vehicles>-<SqFt>; ..."
# Types: G=Garage, C=Carport, R=RV (others observed: B=?, A=?, treat as Other)
# Example: "G2-380;R1-645;C2-420"
_PARK_RE = re.compile(r"^\s*([A-Z]+)(\d+)\s*-\s*(\d+)\s*$")


def parse_park_code(code) -> dict:
    """Parse a single PARK_CODE string into a dict of features."""
    out = {
        "garage_sqft":   0, "garage_spaces":   0,
        "carport_sqft":  0, "carport_spaces":  0,
        "rv_sqft":       0, "rv_spaces":       0,
        "has_garage":    0,
    }
    if not isinstance(code, str) or code == "":
        return out
    for part in code.split(";"):
        m = _PARK_RE.match(part)
        if not m:
            continue
        kind, spaces, sqft = m.group(1), int(m.group(2)), int(m.group(3))
        if kind == "G":
            out["garage_sqft"]  += sqft
            out["garage_spaces"] += spaces
            out["has_garage"] = 1
        elif kind == "C":
            out["carport_sqft"]  += sqft
            out["carport_spaces"] += spaces
        elif kind == "R":
            out["rv_sqft"]  += sqft
            out["rv_spaces"] += spaces
        # other letters: silently ignored
    return out


def add_parking_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    parked = df["PARK_CODE"].apply(parse_park_code).apply(pd.Series)
    df = pd.concat([df, parked], axis=1)
    print(f"  parking parsed: +{parked.shape[1]} columns "
          f"({df['has_garage'].mean()*100:.1f}% have a garage)")
    return df


# ---------------------------------------------------------------------------
# 5. Parse Patios
# ---------------------------------------------------------------------------
# Format: "CV-500;UC-700"   (CV=covered, UC=uncovered)
_PATIO_RE = re.compile(r"^\s*([A-Z]{2})\s*-\s*(\d+)\s*$")


def parse_patios(code) -> dict:
    out = {"covered_patio_sqft": 0, "uncovered_patio_sqft": 0, "has_patio": 0}
    if not isinstance(code, str) or code == "":
        return out
    for part in code.split(";"):
        m = _PATIO_RE.match(part)
        if not m:
            continue
        kind, sqft = m.group(1).upper(), int(m.group(2))
        if kind == "CV":
            out["covered_patio_sqft"] += sqft
            out["has_patio"] = 1
        elif kind == "UC":
            out["uncovered_patio_sqft"] += sqft
            out["has_patio"] = 1
    return out


def add_patio_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    pat = df["Patios"].apply(parse_patios).apply(pd.Series)
    df = pd.concat([df, pat], axis=1)
    print(f"  patios parsed:  +{pat.shape[1]} columns "
          f"({df['has_patio'].mean()*100:.1f}% have a patio)")
    return df


# ---------------------------------------------------------------------------
# 6. Ordinal encode Class
# ---------------------------------------------------------------------------
# Order chosen to match the median price ranking we saw in EDA Q4:
#   LOW MINIMUM ($150k) < R1 ($190k) < R2 < R3 < R4 < R5 < R6 < R7 ($3.2M)
CLASS_ORDER = {
    "LOW MINIMUM": 0,
    "CLASS R1":    1,
    "CLASS R2":    2,
    "CLASS R3":    3,
    "CLASS R4":    4,
    "CLASS R5":    5,
    "CLASS R6":    6,
    "CLASS R7":    7,
}


def encode_class(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["class_ord"] = df["Class"].map(CLASS_ORDER)
    n_unmapped = df["class_ord"].isna().sum()
    if n_unmapped:
        print(f"  WARNING: {n_unmapped} rows had an unknown Class — defaulting to median rank 3")
        df["class_ord"] = df["class_ord"].fillna(3)
    df["class_ord"] = df["class_ord"].astype(int)
    return df


# ---------------------------------------------------------------------------
# 7. One-hot encode small categoricals
# ---------------------------------------------------------------------------
ONE_HOT_COLS = [
    "AirConditioningType",
    "HeatingType",
    "ExteriorWallMaterial",
    "RoofMaterial",
]


def one_hot_encode(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    before = df.shape[1]
    df = pd.get_dummies(df, columns=ONE_HOT_COLS, prefix=ONE_HOT_COLS,
                        dummy_na=False, dtype=int)
    print(f"  one-hot:        +{df.shape[1] - before} columns from {len(ONE_HOT_COLS)} source cols")
    return df


# ---------------------------------------------------------------------------
# 8. Bucket PUC: keep top 20 by count, lump rest as 'OTHER'
# ---------------------------------------------------------------------------
def bucket_puc(df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """Keep the top_n most common PUC codes; lump the rest as 'OTHER'.

    PUC is stored as int64 (codes like 131, 175). We cast to string first so
    the bucketed column has a single consistent dtype (str). Otherwise mixing
    ints and the string 'OTHER' creates an object column that Parquet refuses.
    """
    df = df.copy()
    puc_str = df["PUC"].astype(str)
    top_codes = puc_str.value_counts().head(top_n).index
    df["PUC_bucketed"] = puc_str.where(puc_str.isin(top_codes), other="OTHER")
    n_other = (df["PUC_bucketed"] == "OTHER").sum()
    print(f"  PUC bucketed:   kept top {top_n}, lumped {n_other:,} rows as OTHER")
    return df


# ---------------------------------------------------------------------------
# 9. Final selection — drop the now-unneeded raw columns
# ---------------------------------------------------------------------------
DROP_AFTER_ENGINEER = [
    "PARK_CODE",        # parsed into garage/carport/rv features
    "Patios",           # parsed into patio features
    "Class",            # encoded as class_ord
    "PUC",              # replaced by PUC_bucketed (we keep BOOK as-is)
]


def finalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.drop(columns=DROP_AFTER_ENGINEER, errors="ignore")
    print(f"  finalize:       dropped {len(DROP_AFTER_ENGINEER)} raw cols replaced by engineered ones")
    print(f"  final shape:    {df.shape}")
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    df = load()

    df = drop_and_clip(df)
    df = engineer_numerics(df)
    df = add_parking_features(df)
    df = add_patio_features(df)
    df = encode_class(df)
    df = one_hot_encode(df)
    df = bucket_puc(df)
    df = finalize(df)

    # Quick health check on the final frame
    print("\n=== FINAL HEALTH CHECK ===")
    print(f"  rows: {len(df):,}")
    print(f"  cols: {df.shape[1]}")
    print(f"  dtypes:\n{df.dtypes.value_counts().to_string()}")
    print(f"  any nulls? {df.isna().any().any()}")
    if df.isna().any().any():
        null_cols = df.isna().sum()
        print(null_cols[null_cols > 0].to_string())

    print("\n=== TARGET (log_price) SUMMARY ===")
    print(df["log_price"].describe().apply(lambda x: f"{x:.3f}"))

    out_path = PROCESSED / "model_ready.parquet"
    df.to_parquet(out_path, index=False)
    print(f"\nSaved {len(df):,} rows × {df.shape[1]} cols to {out_path}")
    print(f"  ({out_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
