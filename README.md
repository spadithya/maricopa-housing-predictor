# Maricopa Housing Predictor

An end-to-end machine-learning project: a Streamlit app that predicts
residential property sale prices in Maricopa County, Arizona using public
Assessor data.

> Companion to [Maricopa Housing Stats](../maricopa-housing-stats), which
> provides interactive exploration of the same dataset.

## Live demo

`https://spadida-maricopa-housing-stats.streamlit.app/` 
`https://spadida-maricopa-housing-predictor.streamlit.app/`

Locally:
```bash
streamlit run app.py
```

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate           # Windows
# source .venv/bin/activate      # macOS / Linux

pip install -r requirements.txt
```

### Train from scratch

```bash
# 1. Download Maricopa Assessor R116 export → data/raw/Residential_Master/
# 2. Run the pipeline:
python notebooks/01_get_data.py       # raw → residential_with_sales.parquet
python notebooks/02_clean_features.py # → model_ready.parquet
python notebooks/03_baseline_model.py # OLS + Ridge baselines
python notebooks/04_better_models.py  # Random Forest + XGBoost (winner)
# Artifacts land in models/

# 3. Launch the app
streamlit run app.py
```

## Folder layout

```
maricopa-housing-predictor/
├── app.py                       # Streamlit predictor (entry point)
├── notebooks/                   # Numbered training pipeline
│   ├── 01_get_data.py
│   ├── 02_clean_features.py
│   ├── 03_baseline_model.py
│   └── 04_better_models.py
├── data/
│   ├── raw/                     # Assessor download (gitignored)
│   └── processed/               # Parquet (gitignored)
├── models/                      # Trained artifacts (gitignored)
│   ├── random_forest.joblib
│   ├── xgboost.joblib
│   ├── scaler.joblib
│   └── preprocess.joblib
├── src/                         # Reusable helpers
├── requirements.txt
├── README.md
└── LICENSE
```

## Model performance (test set, n = 91,269)

| Model                  | log MAE | log R² | $ MAE     | MAPE   |
|------------------------|---------|--------|-----------|--------|
| Baseline (predict mean)| 0.4040  | 0.000  | $216,089  | 42.83% |
| OLS (Phase 3)          | 0.1921  | 0.705  | $111,234  | 20.14% |
| Ridge α=1.0            | 0.1921  | 0.705  | $111,232  | 20.14% |
| **Random Forest**      | **0.1226** | **0.846** | **$73,771** | **13.14%** |
| XGBoost                | 0.1330  | 0.833  | $79,649   | 13.89% |

## Notes on the data

- The Assessor's delivered file is pipe-delimited, 24 columns, no header row.
- We filter to sale years 2018-2026 and prices in $50K-$5M.
- BOOK (first 3 digits of APN) substitutes for ZIP, since address columns
  weren't shipped in our extract.
- See the [companion stats project](../maricopa-housing-stats) for full EDA.

## License

Code is MIT-licensed ([LICENSE](LICENSE)).
Data belongs to the Maricopa County Assessor's Office and is not redistributed
here.
