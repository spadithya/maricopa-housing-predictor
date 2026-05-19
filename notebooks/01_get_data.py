"""
Phase 1 — Get the data
======================

Goal: Load whatever raw file you've dropped into ../data/raw/ and inspect it
enough to know what we're working with.

Run this script after you've placed a CSV (or ZIP/TXT from the Assessor)
in ../data/raw/. It will:
  1. Find the largest data file in data/raw/
  2. Load it (handles CSV, TSV, fixed-width, ZIP'd CSV)
  3. Print shape, dtypes, missingness, and a head() preview
  4. Save a 1000-row sample to data/processed/sample_for_inspection.csv
     so we can browse it in Excel without loading the full file each time

Why this matters: before any modeling, we need to know
  - How many rows / columns
  - What the column names mean (need to match against a data dictionary)
  - Where missing values are concentrated
  - Whether sale prices are even present, or just assessed values
"""

from pathlib import Path
import zipfile
import pandas as pd

# ---------------------------------------------------------------------------
# Paths — using pathlib so it works on Windows + Linux
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Schema — from data/raw/Legend.xlsx (Maricopa Assessor R116 Residential Master)
# The delivered file has columns 1-24 only (owner/situs fields 25-39 are absent).
# ---------------------------------------------------------------------------
COLUMN_NAMES = [
    "PARCELNUMBER",         # 1  9-char APN, structured BOOK(3)+MAP(2)+ITEM(3-4)
    "ProportionComplete",   # 2  0.00..1.00
    "Class",                # 3  Property class code (e.g. R3)
    "StoryCount",           # 4
    "AirConditioningType",  # 5
    "HeatingType",          # 6
    "BathroomFixtures",     # 7  Count
    "ExteriorWallMaterial", # 8
    "RoofMaterial",         # 9
    "RoofStyle",            # 10
    "ConstructionYear",     # 11
    "Living_sqft",          # 12
    "FirstFloor_sqft",      # 13
    "SecondFloor_sqft",     # 14
    "ThirdFloor_sqft",      # 15
    "Basement_sqft",        # 16
    "PARK_CODE",            # 17  e.g. "G2-380;R1-645" (garage, RV...)
    "Patios",               # 18  e.g. "CV-500;UC-700"
    "POOL_SQFT",            # 19
    "SALE_PRICE",           # 20  Target
    "SALE_DATE",            # 21  MM/DD/YYYY
    "ADDED_SQFT",           # 22
    "DETACH_SQFT",          # 23
    "PUC",                  # 24  Property Use Code
]
assert len(COLUMN_NAMES) == 24


# ---------------------------------------------------------------------------
# Filtering — what counts as a "real" recent sale we can train on
# ---------------------------------------------------------------------------
MIN_SALE_PRICE = 50_000      # cut out $1/$10 quit-claim transfers, family gifts, etc.
MAX_SALE_PRICE = 5_000_000   # cut out the rare ultra-luxury outliers that ruin RMSE
MIN_SALE_YEAR  = 2018        # recent enough that prices reflect today's market


# ---------------------------------------------------------------------------
# Step 1: Find the file
# ---------------------------------------------------------------------------
def find_largest_data_file(raw_dir: Path) -> Path:
    """Return the largest data-ish file anywhere under raw_dir (recursive).

    Assessor downloads often ship as a folder tree:
        raw/Residential_Master/Data/<the actual file>
        raw/Residential_Master/FileSpec/<schema PDF / TXT>
    so we recurse with rglob and skip the FileSpec folder so we don't pick up
    the data dictionary instead of the data itself.
    """
    DATA_EXTS = {".csv", ".tsv", ".txt", ".zip", ".xlsx", ".dat"}
    candidates = [
        p for p in raw_dir.rglob("*")
        if p.is_file()
        and p.suffix.lower() in DATA_EXTS
        and "FileSpec" not in p.parts          # skip the schema folder
        and "sample_for_inspection" not in p.name
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No data files anywhere under {raw_dir}. "
            "Drop a CSV/ZIP/TXT from the Assessor (or Kaggle) into that folder."
        )
    chosen = max(candidates, key=lambda p: p.stat().st_size)
    print(f"Found {len(candidates)} candidate file(s); picking largest: {chosen}")
    return chosen


def find_filespec(raw_dir: Path) -> list[Path]:
    """Locate any schema/dictionary files (FileSpec folder contents)."""
    return [p for p in raw_dir.rglob("*") if p.is_file() and "FileSpec" in p.parts]


# ---------------------------------------------------------------------------
# Step 2: Load it (be permissive — Assessor files often have weird delimiters)
# ---------------------------------------------------------------------------
def load_any(path: Path, names: list[str] | None = None) -> pd.DataFrame:
    """Try a few common formats. If `names` is provided, read with header=None
    and assign those column names — needed because the Assessor file ships
    without a header row."""
    suffix = path.suffix.lower()

    if suffix == ".zip":
        with zipfile.ZipFile(path) as z:
            members = [n for n in z.namelist() if n.lower().endswith((".csv", ".txt", ".tsv"))]
            if not members:
                raise ValueError(f"No CSV-like files inside {path.name}")
            inner = members[0]
            print(f"Reading {inner} from inside {path.name}")
            with z.open(inner) as f:
                return _read_delimited(f, inner, names=names)

    if suffix == ".xlsx":
        return pd.read_excel(path)

    return _read_delimited(path, str(path), names=names)


def _read_delimited(path_or_buf, hint_name: str, names: list[str] | None = None) -> pd.DataFrame:
    """Sniff the delimiter from a small sample, then read once with the winner.

    Why sniff? On big files (200+ MB) we don't want to read the whole thing
    three times to figure out the delimiter. We peek at the first 4 KB,
    count occurrences of common delimiters, and pick the most frequent.
    """
    LABELS = {"|": "pipe", "\t": "tab", ",": "comma"}

    # 1. Get a small text sample without consuming the file
    if hasattr(path_or_buf, "read"):
        sample = path_or_buf.read(4096)
        if isinstance(sample, bytes):
            sample = sample.decode("utf-8", errors="replace")
        path_or_buf.seek(0)
    else:
        with open(path_or_buf, "r", encoding="utf-8", errors="replace") as fh:
            sample = fh.read(4096)

    # 2. Pick the delimiter that appears most often in the sample
    counts = {sep: sample.count(sep) for sep in LABELS}
    best_sep = max(counts, key=counts.get)
    if counts[best_sep] == 0:
        raise ValueError(
            f"No common delimiter found in the first 4 KB of {hint_name}. "
            f"Counts: {counts}"
        )
    print(f"Sniffed delimiter for {hint_name}: {LABELS[best_sep]!r}  "
          f"(counts in first 4 KB: { {LABELS[k]: v for k,v in counts.items()} })")

    # 3. Read the file once with the chosen delimiter.
    # If `names` was passed, the file has no header row — tell pandas so.
    read_kwargs = dict(
        sep=best_sep,
        low_memory=False,
        on_bad_lines="warn",
        encoding_errors="replace",
    )
    if names is not None:
        read_kwargs["header"] = None
        read_kwargs["names"] = names
    df = pd.read_csv(path_or_buf, **read_kwargs)
    print(f"Loaded shape: {df.shape}")
    return df


# ---------------------------------------------------------------------------
# Step 3: Inspect
# ---------------------------------------------------------------------------
def inspect(df: pd.DataFrame) -> None:
    print("\n=== SHAPE ===")
    print(df.shape)

    print("\n=== DTYPES ===")
    print(df.dtypes.value_counts())

    print("\n=== FIRST 5 ROWS ===")
    print(df.head())

    print("\n=== COLUMNS ===")
    for col in df.columns:
        print(f"  {col}")

    print("\n=== MISSINGNESS (top 20) ===")
    missing = df.isna().mean().sort_values(ascending=False).head(20)
    print((missing * 100).round(2).astype(str) + " %")

    # Look for likely target columns
    print("\n=== POSSIBLE TARGET COLUMNS ===")
    target_keywords = ["sale", "price", "amount", "consideration", "value"]
    candidates = [c for c in df.columns if any(k in c.lower() for k in target_keywords)]
    for c in candidates:
        nonnull = df[c].notna().sum()
        print(f"  {c}  (non-null: {nonnull:,})")


# ---------------------------------------------------------------------------
# Step 4: Filter to a usable modeling universe + add a geography proxy
# ---------------------------------------------------------------------------
def add_book_feature(df: pd.DataFrame) -> pd.DataFrame:
    """Add BOOK = first 3 digits of PARCELNUMBER (geographic cluster).

    The APN is structured BBB-MM-IIII. Parcels in the same BOOK are
    geographically close, so BOOK is a decent proxy for "neighborhood"
    without addresses. We zero-pad in case some APNs are <9 digits.
    """
    df = df.copy()
    df["PARCELNUMBER"] = df["PARCELNUMBER"].astype(str).str.zfill(9)
    df["BOOK"] = df["PARCELNUMBER"].str[:3]
    return df


def filter_valid_sales(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only rows that represent real, recent market transactions.

    Note: the FileSpec legend claims SALE_DATE is MM/DD/YYYY, but the actual
    delivered file uses ISO YYYY-MM-DD. We let pandas infer the format
    instead of trusting the doc.
    """
    n0 = len(df)
    df = df.copy()

    # Defensive parsing — auto-detect format, NaT for anything unparseable
    df["SALE_DATE"]  = pd.to_datetime(df["SALE_DATE"], errors="coerce")
    df["SALE_PRICE"] = pd.to_numeric(df["SALE_PRICE"], errors="coerce")

    n_dates  = df["SALE_DATE"].notna().sum()
    n_prices = df["SALE_PRICE"].notna().sum()
    print("\n=== PARSING RESULTS ===")
    print(f"  rows with parseable SALE_DATE:  {n_dates:>12,} / {n0:,}")
    print(f"  rows with parseable SALE_PRICE: {n_prices:>12,} / {n0:,}")

    # Year distribution of actual sales (price > 0) — helps us calibrate the cutoff
    has_sale = df[(df["SALE_DATE"].notna()) & (df["SALE_PRICE"] > 0)]
    by_year = has_sale["SALE_DATE"].dt.year.value_counts().sort_index()
    print(f"\n=== SALE YEAR DISTRIBUTION (price > 0), last 15 years ===")
    print(by_year.tail(15).to_string())

    mask_price = df["SALE_PRICE"].between(MIN_SALE_PRICE, MAX_SALE_PRICE)
    mask_date  = df["SALE_DATE"].dt.year >= MIN_SALE_YEAR
    mask_year  = df["ConstructionYear"].between(1850, 2026)  # sanity

    out = df[mask_price & mask_date & mask_year].reset_index(drop=True)

    print("\n=== FILTER FUNNEL ===")
    print(f"  starting rows:                          {n0:>12,}")
    print(f"  after price in [{MIN_SALE_PRICE:,}, {MAX_SALE_PRICE:,}]: "
          f"{mask_price.sum():>12,}")
    print(f"  after sale year >= {MIN_SALE_YEAR}:                 "
          f"{mask_date.sum():>12,}")
    print(f"  after all filters combined:             {len(out):>12,}")
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    spec_files = find_filespec(RAW_DIR)
    if spec_files:
        print("=== FILESPEC / SCHEMA FILES FOUND ===")
        for p in spec_files:
            print(f"  {p}")
        print()

    src = find_largest_data_file(RAW_DIR)
    print(f"Loading: {src}  ({src.stat().st_size / 1e6:.1f} MB)")

    df = load_any(src, names=COLUMN_NAMES)
    inspect(df)

    # Geographic feature from APN structure
    df = add_book_feature(df)
    print(f"\nBOOK feature: {df['BOOK'].nunique():,} distinct books "
          f"(geographic clusters)")

    # Filter to valid recent sales
    sales = filter_valid_sales(df)

    # Quick price summary on the filtered set
    print("\n=== SALE_PRICE SUMMARY (filtered set) ===")
    print(sales["SALE_PRICE"].describe().apply(lambda x: f"{x:,.0f}"))

    # Save the modeling universe as Parquet — fast, small, typed
    out_path = PROCESSED_DIR / "residential_with_sales.parquet"
    sales.to_parquet(out_path, index=False)
    size_mb = out_path.stat().st_size / 1e6
    print(f"\nSaved {len(sales):,} rows to {out_path}  ({size_mb:.1f} MB)")

    # Also save a small sample as CSV for easy Excel browsing
    sample_path = PROCESSED_DIR / "sample_for_inspection.csv"
    sales.head(1000).to_csv(sample_path, index=False)
    print(f"Saved 1000-row sample CSV to {sample_path}")


if __name__ == "__main__":
    main()
