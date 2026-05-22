# Inventory Decision System — Codebase Audit

Static-analysis audit performed without executing any code. Findings cover the production path (raw CSV → snapshot → trained model → FastAPI inference → Streamlit UI) plus all auxiliary scripts.

---

## 1. Directory structure

```
Inventory-Decision-System/
├── Dockerfile                          # Python 3.11-slim image; installs libgomp1, copies api/, src/, data/; runs uvicorn on :8000
├── README.md                           # Project overview, methodology, quickstart, dataset notes
├── PROJECT_TREE.txt                    # Stale 2 MB tree dump from Dec 2024; not consumed by anything
├── Inventory_Decision_System_Interview_Prep.docx  # Interview prep doc, unrelated to runtime
├── requirements.txt                    # Pinned production deps (fastapi, lightgbm, pandas, etc.)
├── requirements_raw.txt                # Old unpinned dep list; gitignored, not used
├── rebuild_pipeline.ps1                # End-to-end rebuild (snapshots + train + activate latest)
├── run_app.ps1                         # Spawns uvicorn + streamlit in two PowerShell windows
├── .gitignore                          # Ignores .venv, raw data, snapshots, deprecated scripts
├── .gitattributes                      # Git LFS rules
│
├── api/
│   ├── __init__.py                     # Empty package marker
│   ├── main.py                         # FastAPI app: /health, /version, /forecast-to-orders
│   └── schemas.py                      # Pydantic request/response models
│
├── src/
│   ├── __init__.py                     # Empty package marker
│   ├── config.py                       # Paths, MODEL_VERSION_LABEL, ACTIVE_DATASET_MODE, snapshot map
│   ├── data/
│   │   ├── sampling.py                 # select_store_item_universe, apply_universe_filter
│   │   ├── snapshot_builder.py         # build_base_snapshot (join train+items+stores, date-filter)
│   │   └── validation.py               # Generic column / dtype / null validators (DEAD — see §5)
│   ├── features/
│   │   ├── __init__.py                 # Empty package marker
│   │   ├── calendar.py                 # add_calendar_features (year, month, dayofweek, weekofyear, is_weekend)
│   │   ├── holidays.py                 # add_holiday_feature (is_holiday from holidays_events.csv)
│   │   ├── oil.py                      # add_oil_feature (dcoilwtico, ffill+bfill)
│   │   ├── promotion.py                # add_promotion_feature (onpromotion → 0/1 int)
│   │   ├── lags.py                     # add_lag_features (lag_7/14/28, rolling_7/14)
│   │   ├── categorical.py              # extract/apply/save/load category schemas (partly DEAD — see §5)
│   │   └── feature_pipeline.py         # apply_all_features (DEAD — never called)
│   ├── ml/
│   │   ├── __init__.py                 # Empty package marker
│   │   ├── feature_config.py           # FEATURES, CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET_COL
│   │   ├── splits.py                   # TRAIN_START/END, VALID_START/END date constants
│   │   ├── trainer.py                  # train_lgbm_quantile (LightGBM quantile training)
│   │   ├── predictor.py                # ModelRegistry + QuantilePredictor (load, schema apply, predict, expm1)
│   │   └── predictor_factory.py        # build_predictor(version) / build_default_predictor()
│   ├── optimization/
│   │   └── optimizer.py                # optimize_proportional_allocation (capacity-capped largest-remainder)
│   ├── utils/
│   │   └── __init__.py                 # Empty; entire package is empty
│   └── validation/
│       ├── feature_validation.py       # validate_base_snapshot, validate_featured_snapshot
│       └── schema_checks.py            # check_required_columns (DEAD — see §5)
│
├── ui/
│   ├── app.py                          # Streamlit client; uploads SKU CSV, calls API, capacity stress sweep, plot
│   └── test_payload.csv                # 8-row demo SKU list (item_nbr, onpromotion)
│
├── scripts/
│   ├── build_training_snapshot.py      # raw train.csv → universe filter → base train snapshot parquet
│   ├── build_featured_snapshot.py      # base train snapshot → featured train snapshot parquet
│   ├── build_test_snapshot_2016Q1.py   # raw train.csv → 2016 Feb–Apr base test snapshot parquet
│   ├── build_test_featured_snapshot_2016Q1.py  # base test snapshot → featured test snapshot parquet
│   ├── train_quantile_model.py         # Train P90/P95 LGBM, save schemas, write metadata, optionally update latest
│   ├── evaluate_quantile_calibration_2016Q1.py # Coverage + pinball loss on 2016Q1 test snapshot
│   ├── validate_quantile_model.py      # Coverage + pinball loss on 2015 validation window
│   ├── debug_pick_valid_request.py     # Picks a (store, date) with many SKUs and prints an API payload
│   ├── dump_repo_tree.py               # Prints an ASCII tree of the repo (utility)
│   ├── demos/
│   │   ├── demo_forecast_to_orders.py  # Standalone end-to-end demo (forecast + optimize, no API)
│   │   ├── demo_risk_knob.py           # Compares P90 vs P95 distributions on a 5k sample
│   │   ├── run_inference_sample.py     # Inference sanity check on 5k random rows
│   │   └── scenario_comparison.py      # Sweep service_level × floor_ratio × perishable_weight
│   └── deprecated/                     # gitignored; not part of runtime
│       ├── train_models.py             # Earlier (un-versioned) training script — broken (see §6)
│       ├── test_feature_pipeline.py    # Imports a non-existent function (see §4)
│       ├── apply_sampling_universe.py  # Imports non-existent functions (see §4)
│       ├── save_category_schemas.py    # Pre-versioning schema dump
│       ├── add_calendar_features.py    # Pre-pipeline calendar step
│       ├── check_data_layout.py        # One-off raw CSV sanity print
│       └── sanity_check.py             # One-off feature/target presence check
│
├── data/
│   ├── raw/                            # Kaggle Favorita CSVs (train, items, stores, oil, holidays_events, transactions, test, sample_submission)
│   ├── processed/                      # Empty
│   ├── snapshots/
│   │   ├── favorita_train_snapshot_2015.parquet           # Base training snapshot file
│   │   ├── favorita_train_snapshot_2015/                  # ALSO a directory of the same stem — see §6
│   │   ├── favorita_train_featured_2015.parquet           # Featured training snapshot
│   │   ├── favorita_test_snapshot_2016Q1.parquet          # Base 2016 test snapshot
│   │   ├── favorita_test_featured_2016Q1.parquet          # Featured 2016 test snapshot (served by API)
│   │   └── experimental/                                  # Old intermediate artifacts (gitignored)
│   └── models/
│       ├── latest/                     # Active model artifacts (loaded by API)
│       │   ├── favorita_lgbm_p90.txt
│       │   ├── favorita_lgbm_p95.txt
│       │   ├── category_schemas.json
│       │   └── metadata.json           # Reports version "v_2026_05_22"
│       ├── v_2026_05_22/               # Same artifacts; source for latest/
│       └── v1/                         # Older model version, kept on disk
│
└── notebooks/                          # Project_Favorita_V*.ipynb exploration notebooks (untracked)
```

No `tests/` directory exists at the project root.

---

## 2. Data flow — raw CSV to served API response

### Training path (offline, run via `rebuild_pipeline.ps1`)

1. **`scripts/build_training_snapshot.py`**
   - Reads `data/raw/train.csv` with columns `[date, store_nbr, item_nbr, unit_sales, onpromotion]`.
   - `select_store_item_universe(train, top_n_stores=25, min_item_obs=500, top_n_items=800)` → universe of stores and items.
   - `apply_universe_filter(train, stores, items)` → row-filtered train.
   - Reads `data/raw/items.csv`, `data/raw/stores.csv`.
   - `build_base_snapshot(train, items, stores, START_DATE="2013-01-01", END_DATE="2015-12-31")` joins dimensions and date-filters.
   - Writes `data/snapshots/favorita_train_snapshot_2015.parquet`.

2. **`scripts/build_featured_snapshot.py`**
   - Reads `data/snapshots/favorita_train_snapshot_2015.parquet`.
   - `validate_base_snapshot(df)`.
   - Reads `data/raw/holidays_events.csv` (date, description), `data/raw/oil.csv` (date, dcoilwtico).
   - Applies: `add_calendar_features` → `add_holiday_feature` → `add_oil_feature` → `add_promotion_feature` → sort by (store, item, date) → `add_lag_features(lags=[7,14,28], rolls=[7,14])`.
   - `validate_featured_snapshot(df)` (lag/rolling NaNs allowed; non-lag features must be non-null).
   - Writes `data/snapshots/favorita_train_featured_2015.parquet`.

3. **`scripts/build_test_snapshot_2016Q1.py`**
   - Reads full `data/raw/train.csv` again.
   - `select_store_item_universe(train_df)` — recomputed from raw train, default args; matches training universe.
   - Filters rows to `2016-02-01 ≤ date ≤ 2016-04-30` AND universe membership.
   - `build_base_snapshot(df_slice, items, stores, START_DATE, END_DATE)` — note `build_base_snapshot` also re-applies the same date filter.
   - Writes `data/snapshots/favorita_test_snapshot_2016Q1.parquet`.
   - **Flag:** No January 2016 rows are loaded, but `add_lag_features` later needs the prior 28 days for the Feb 1 row. See §6 bug 1.

4. **`scripts/build_test_featured_snapshot_2016Q1.py`**
   - Reads `data/snapshots/favorita_test_snapshot_2016Q1.parquet`.
   - `validate_base_snapshot(df)`.
   - Reads `holidays_events.csv` and `oil.csv` (no `usecols`, all columns).
   - Same feature steps as build_featured_snapshot.py, inlined again.
   - `validate_featured_snapshot(df)`.
   - Writes `data/snapshots/favorita_test_featured_2016Q1.parquet`.

5. **`scripts/train_quantile_model.py`**
   - Reads `favorita_train_featured_2015.parquet`.
   - Casts `date` to `dt.date` for `<=` comparisons against `datetime.date` constants in `src/ml/splits.py`.
   - Splits: train = `2013-01-01..2015-06-30`, valid = `2015-07-01..2015-12-31`.
   - `extract_category_schemas(train_df, CATEGORICAL_FEATURES)` → JSON saved to `data/models/<version>/category_schemas.json`. **Schemas come from TRAIN ONLY** (not from the test/validation slice).
   - For each quantile in `[0.90, 0.95]`: `train_lgbm_quantile(...)` saves `favorita_lgbm_p{90,95}.txt`.
   - Writes `metadata.json` with version, trained_at, dataset, windows, quantiles.
   - If `--update-latest`, `rmtree` + `copytree` to `data/models/latest/`.

### Inference path (online, FastAPI)

6. **`api/main.py` startup**
   - `build_default_predictor()` → loads `data/models/latest/category_schemas.json`, lazily wires `favorita_lgbm_p90.txt` and `favorita_lgbm_p95.txt`.
   - Reads `data/snapshots/<FEATURED_SNAPSHOT_BY_MODE[ACTIVE_DATASET_MODE]>` once. With `ACTIVE_DATASET_MODE="test"`, this is `favorita_test_featured_2016Q1.parquet`.
   - Coerces `date` column to datetime64.

7. **`POST /forecast-to-orders`**
   - Parses `req.date` → Timestamp.
   - Slices: `df_features[(store_nbr == req.store_nbr) & (date == decision_date)]`.
   - 404 if empty.
   - Filters to `item_nbr in request.items`. 404 if empty.
   - `sort_values("item_nbr") → drop_duplicates(subset=["item_nbr"], keep="last")` to enforce one row per SKU.
   - Overrides `onpromotion` column from request payload (uses `req` value, not snapshot value).
   - `predictor.predict_df(df_slice, service_level=req.service_level)`:
     - `_apply_category_schemas` (training categories enforced, unseen → NaN).
     - Validates FEATURES presence, selects columns, calls `model.predict`, applies `np.expm1`, clips negatives.
   - Builds `demand = {item_nbr: forecast}` and `perishable_flags = {item_nbr: perishable_int}`.
   - `optimize_proportional_allocation(demand, capacity, service_floor_ratio, perishable_flags, perishable_weight, fill_capacity=False)`:
     - Effective capacity = `min(capacity, total_weighted_demand)`.
     - Apply (possibly scaled) per-SKU floors.
     - Proportionally distribute residual capacity.
     - Round with largest-remainder.
   - Builds response: per-SKU `{item_nbr, forecast, order_qty}` plus summary totals and metadata.

### UI

8. **`ui/app.py`**
   - On launch, polls `GET /health` and `GET /version`.
   - User uploads/loads a CSV with `(item_nbr, onpromotion)`.
   - On "Run", posts request → renders order table.
   - Capacity stress test: 30 additional POSTs at scaled capacities; plots a step coverage curve and finds the minimum capacity meeting the slider target.

### Items flagged as broken or unclear in this flow

- **`build_test_snapshot_2016Q1.py` start at Feb 1 with no Jan history loaded.** Lag_28 on Feb 1 will be NaN. README's claim "starts in February 2016 ... to ensure all 28-day lag features are fully populated from January history" is not enforced by the code. (See §6 bug 1.)
- **`api/main.py` startup comment** "SAFE: never crashes on missing latest/" is incorrect — `build_default_predictor()` raises `FileNotFoundError` if `latest/` is missing.
- **`api/main.py` does not recompute lag/calendar features per request.** It relies entirely on whatever lag values the snapshot was built with. Therefore, after a model retrain the snapshot must be regenerated and the API restarted for changes to take effect.
- **`api/main.py`** silently drops items the snapshot does not contain (e.g., item_nbr outside the top-800 universe, or no row for the given store/date). Only matching items appear in the response; the caller is not told which were dropped.
- **`MODEL_VERSION_LABEL = "v1"`** in `src/config.py` is stale; `latest/metadata.json` reports `v_2026_05_22`. `/version` returns `"v1"` regardless.

---

## 3. Feature consistency check

`src/ml/feature_config.py` defines:

```
CATEGORICAL_FEATURES = [family, class, city, state, type, cluster]
NUMERIC_FEATURES     = [year, month, weekofyear, dayofweek,
                        onpromotion, is_weekend, is_holiday, perishable,
                        dcoilwtico,
                        lag_7, lag_14, lag_28, rolling_7, rolling_14]
FEATURES             = CATEGORICAL_FEATURES + NUMERIC_FEATURES   # 20 features
```

### Where each feature originates in the featured-snapshot builders

| Feature | Source step |
|---|---|
| family, class, perishable | `build_base_snapshot` join with `items.csv` |
| city, state, type, cluster | `build_base_snapshot` join with `stores.csv` |
| year, month, weekofyear, dayofweek, is_weekend | `add_calendar_features` |
| is_holiday | `add_holiday_feature` |
| dcoilwtico | `add_oil_feature` |
| onpromotion | `add_promotion_feature` (overridden at request time in API) |
| lag_7, lag_14, lag_28, rolling_7, rolling_14 | `add_lag_features` with `lags=[7,14,28], rolls=[7,14]` |

### Training-snapshot builder (`scripts/build_featured_snapshot.py`)

- LAGS = [7, 14, 28], ROLLS = [7, 14]. ✓ matches feature_config.
- Applies all five feature functions in order (calendar → holiday → oil → promotion → lags). ✓
- Produces every name in `FEATURES`. ✓

### Test-snapshot builder (`scripts/build_test_featured_snapshot_2016Q1.py`)

- Hard-codes `lags=[7,14,28], rolls=[7,14]`. ✓ matches.
- Applies the same five steps in the same order. ✓
- Produces every name in `FEATURES`. ✓

### API inference (`api/main.py`)

- Does not recompute features. Reads the pre-built featured snapshot.
- Overrides only `onpromotion` from request items.
- Passes the row(s) to `predictor.predict_df`, which selects exactly `FEATURES` from `feature_config.py`.

**Conclusion:** the feature set defined in `feature_config.py` is produced by both featured-snapshot builders and consumed by the predictor. **No feature-list discrepancies detected** in the production path.

### Side notes (not discrepancies, but worth recording)

- `src/features/feature_pipeline.apply_all_features` exists as a single-call orchestrator for all five feature steps, but neither `build_featured_snapshot.py` nor `build_test_featured_snapshot_2016Q1.py` uses it — both inline the same five calls. This duplicates logic but does not break consistency. See §5.
- The training script's `extract_category_schemas` is called on the **train slice only** (post date-split). Categories present in valid/test but not in train will be mapped to NaN by `_apply_category_schemas` in `predictor.py`. This is correct and intentional; documented here so the boundary is visible.

---

## 4. Function signature audit

Every `from src.* import …` statement in the production path and active scripts was checked against the definition site.

### Active code — all match

| Import | Defined |
|---|---|
| `from src.config import RAW_DIR, PROCESSED_DIR, SNAPSHOTS_DIR, MODELS_DIR, MODEL_VERSION_LABEL, ACTIVE_DATASET_MODE, FEATURED_SNAPSHOT_BY_MODE` | `src/config.py` ✓ |
| `from src.data.snapshot_builder import build_base_snapshot` | `src/data/snapshot_builder.py:5` ✓ |
| `from src.data.sampling import select_store_item_universe, apply_universe_filter` | `src/data/sampling.py:5, 44` ✓ |
| `from src.features.calendar import add_calendar_features` | `src/features/calendar.py:4` ✓ |
| `from src.features.holidays import add_holiday_feature` | `src/features/holidays.py:4` ✓ |
| `from src.features.oil import add_oil_feature` | `src/features/oil.py:4` ✓ |
| `from src.features.promotion import add_promotion_feature` | `src/features/promotion.py:4` ✓ |
| `from src.features.lags import add_lag_features` | `src/features/lags.py:5` ✓ |
| `from src.features.categorical import extract_category_schemas, save_category_schemas` | `src/features/categorical.py:7, 37` ✓ |
| `from src.validation.feature_validation import validate_base_snapshot, validate_featured_snapshot` | `src/validation/feature_validation.py:17, 79` ✓ |
| `from src.ml.feature_config import FEATURES, TARGET_COL, CATEGORICAL_FEATURES` | `src/ml/feature_config.py:3, 12, 45` ✓ |
| `from src.ml.splits import TRAIN_START, TRAIN_END, VALID_START, VALID_END` | `src/ml/splits.py:5–9` ✓ |
| `from src.ml.trainer import train_lgbm_quantile` | `src/ml/trainer.py:8` ✓ |
| `from src.ml.predictor import ModelRegistry, QuantilePredictor` | `src/ml/predictor.py:13, 22` ✓ |
| `from src.ml.predictor_factory import build_predictor, build_default_predictor` | `src/ml/predictor_factory.py:8, 41` ✓ |
| `from src.optimization.optimizer import optimize_proportional_allocation` | `src/optimization/optimizer.py:5` ✓ |
| `from api.schemas import ForecastToOrdersRequest, ForecastToOrdersResponse` | `api/schemas.py:22, 87` ✓ |

### Deprecated scripts — mismatches present (all under `scripts/deprecated/`, which is gitignored)

- **`scripts/deprecated/test_feature_pipeline.py:5`**
  `from src.data.snapshot_builder import build_training_snapshot` — no `build_training_snapshot` exists in `src/data/snapshot_builder.py`; the actual name is `build_base_snapshot`. **Will raise `ImportError`.**

- **`scripts/deprecated/apply_sampling_universe.py:3`**
  `from src.data.sampling import load_sampling_universe, apply_sampling_universe` — neither function exists. `src/data/sampling.py` defines `select_store_item_universe` and `apply_universe_filter`. **Will raise `ImportError`.**

No mismatches were found in any non-deprecated module.

---

## 5. Dead code

### Files / packages

- **`src/utils/`** — contains only an empty `__init__.py`. No symbols, no imports of it.
- **`PROJECT_TREE.txt`** (2 MB) and **`requirements_raw.txt`** — not referenced by any script or doc.
- **`Inventory_Decision_System_Interview_Prep.docx`** — personal interview prep; not consumed by code.
- **`scripts/deprecated/`** — entire directory, gitignored, not in the runtime path. Includes `train_models.py` (references a snapshot name that does not exist, `favorita_train_model_table_2015.parquet`), `test_feature_pipeline.py`, `apply_sampling_universe.py`, `save_category_schemas.py`, `add_calendar_features.py`, `check_data_layout.py`, `sanity_check.py`.
- **`data/snapshots/experimental/`** — old intermediate parquets (`*_cal.parquet`, `*_cal_sampled.parquet`, `*_oil_holiday.parquet`, `sample_items/stores.parquet`); not read by anything in the current codebase.
- **`data/models/v1/`** — superseded by `v_2026_05_22` / `latest/`.

### Functions

- **`src/features/feature_pipeline.apply_all_features`** — never called. The two featured-snapshot builders inline the same five steps independently. This is the canonical orchestrator and is the natural place to deduplicate the two builders' feature blocks, but currently it is unused.
- **`src/data/validation.py`** — `validate_required_columns`, `validate_dtypes`, `validate_missingness` — none of these are imported anywhere. (Validation is instead done by `src/validation/feature_validation.py`.)
- **`src/validation/schema_checks.py`** — `check_required_columns` — never imported.
- **`src/features/categorical.py`** — `apply_category_schemas` and `load_category_schemas` are never imported. `QuantilePredictor._apply_category_schemas` and the inline `json.load` in `predictor.py:32–33` re-implement what these helpers offer. Only `extract_category_schemas` and `save_category_schemas` are used (by `train_quantile_model.py`).
- **`api/main.py:121–122`** — `if not df_slice["item_nbr"].is_unique` check is unreachable because `drop_duplicates(subset=["item_nbr"])` was applied immediately above.
- **`scripts/dump_repo_tree.py`** — utility, not part of any pipeline.
- **`scripts/debug_pick_valid_request.py`** — interactive helper.
- **`scripts/evaluate_quantile_calibration_2016Q1.py`** and **`scripts/validate_quantile_model.py`** — useful evaluation scripts but not invoked by `rebuild_pipeline.ps1` or the API; they are standalone.
- **`scripts/demos/*`** — four standalone demo scripts (forecast/orders, risk knob, sample inference, scenario comparison). None are called by the pipeline or the app.

### Unused imports

- **`api/main.py:3`** — `import pandas as pd` is used (✓); `import numpy as np` is used (✓). No unused imports here.
- **`scripts/train_quantile_model.py:3`** — `import shutil` is used (✓); `import json` is used (✓).
- **`scripts/build_test_snapshot_2016Q1.py:2`** — `from pathlib import Path` is imported but never used.
- **`scripts/build_test_featured_snapshot_2016Q1.py:2`** — `from pathlib import Path` is imported but never used.
- **`scripts/evaluate_quantile_calibration_2016Q1.py:3`** — `from pathlib import Path` is imported but never used.
- **`api/schemas.py`** — all symbols used by `api/main.py`.

---

## 6. Bugs and inconsistencies

### Bug 1 — Test snapshot is missing the prior-month history needed for its own lag features

`scripts/build_test_snapshot_2016Q1.py` filters rows to `2016-02-01 ≤ date ≤ 2016-04-30` **before** the featured snapshot is built. `scripts/build_test_featured_snapshot_2016Q1.py` then computes `lag_7/14/28` and `rolling_7/14` on that base snapshot. Because there is no January 2016 (or earlier) data in the base snapshot, every Feb 1–28 row will have NaN `lag_28`, and most early February rows will also have NaN `lag_7`, `lag_14`, `rolling_7`, `rolling_14`.

`validate_featured_snapshot` explicitly allows lag/rolling NaNs, so it does not catch this. LightGBM handles NaN natively, so the API does not crash — it just serves degraded forecasts for early dates. This **directly contradicts the README**:

> "The deployment test set starts in February 2016 rather than January to ensure all 28-day lag features are fully populated from January history."

Either:
- the README's claim is wrong, and the snapshot was deliberately built without prior history; or
- the snapshot was meant to load earlier history (e.g., `2016-01-04` onward) and trim back to `>= 2016-02-01` only *after* lag computation.

### Bug 2 — Test snapshot file name says "2016Q1" but spans Feb–Apr

`START_DATE = "2016-02-01"`, `END_DATE = "2016-04-30"`. Calendar Q1 is January–March. The artifact `favorita_test_snapshot_2016Q1.parquet` is misnamed (it's really Feb–Apr 2016).

### Bug 3 — `MODEL_VERSION_LABEL` is stale in `src/config.py`

```
MODEL_VERSION_LABEL = "v1"
```

but `data/models/latest/metadata.json` reports the active version as `v_2026_05_22`. `GET /version` returns the stale `"v1"` label, which is misleading. The same constant is used by `scripts/demos/demo_forecast_to_orders.py` and `scripts/demos/run_inference_sample.py` to construct a model path (`MODELS_DIR / "v1"`); those scripts therefore load the **old** v1 artifacts, not the current `latest`.

### Bug 4 — Misleading "SAFE" comment in `api/main.py`

```
# Load predictor (SAFE: never crashes on missing latest/)
predictor = build_default_predictor()
```

`build_default_predictor` → `build_predictor(None)` → raises `FileNotFoundError` if `data/models/latest/` is absent. The comment misrepresents the behavior.

### Bug 5 — Unreachable defensive check in `api/main.py`

After `drop_duplicates(subset=["item_nbr"], keep="last")`, the immediately following `if not df_slice["item_nbr"].is_unique: raise HTTPException(...)` cannot trigger. Dead defensively but worth removing for clarity.

### Bug 6 — Streamlit can throw `StopIteration` if target is unreachable

`ui/app.py:295–297`:

```
required_capacity = next(
    cap for cap, cov in zip(cap_grid, coverages) if cov >= target
)
```

If the user slides the target above the maximum coverage observed on the sweep, `next(...)` raises `StopIteration`, which Streamlit will surface as an uncaught error. Should use `next(..., None)` and render a message.

### Bug 7 — `validate_dtypes` uses deprecated `is_categorical_dtype`

`src/data/validation.py:33` calls `pd.api.types.is_categorical_dtype`, which is deprecated in pandas 2.1 and removed in 2.2. Pinned pandas is 2.3.3, so importing the function still works (deprecated alias), but if it is ever called it emits a warning. Function is dead anyway (see §5), but flag for completeness.

### Bug 8 — Two artifacts share the stem `favorita_train_snapshot_2015`

`data/snapshots/` contains both `favorita_train_snapshot_2015` (directory) and `favorita_train_snapshot_2015.parquet` (file). `to_parquet(...)` with a directory-style path produces partitioned writes; with a `.parquet` suffix it writes a single file. Having both suggests an aborted prior write. Anything that does `pd.read_parquet(SNAPSHOTS_DIR / "favorita_train_snapshot_2015.parquet")` reads the file, but the bare-stem directory is leftover state that could confuse future runs.

### Bug 9 — `train_quantile_model.py` casts dates to `dt.date` solely for split comparison

```
df["date"] = pd.to_datetime(df["date"]).dt.date
train_df = df[(df["date"] >= TRAIN_START) & (df["date"] <= TRAIN_END)]
```

This converts the column from `datetime64[ns]` to a Python-object column of `date` instances, then compares element-wise against `datetime.date` constants. It works, but downstream the `date` column inside `train_df` and `valid_df` is now object-dtype. `date` is not in `FEATURES`, so this does not corrupt training, but anything else that consumes these frames must tolerate the dtype change.

### Bug 10 — `train_quantile_model.py` uses `model_dir.mkdir(exist_ok=False)`

`exist_ok=False` means rerunning the script with the same `--version` raises `FileExistsError`. Intentional (forces fresh versions), but if combined with `--update-latest` this prevents idempotent reruns. Worth flagging because `rebuild_pipeline.ps1` defaults to `v_$(Get-Date -Format 'yyyy_MM_dd')` and will fail on the second same-day run.

### Bug 11 — `optimize_proportional_allocation` largest-remainder rounding can over-allocate

```
gap = int(round(effective_capacity - current_total))
```

`effective_capacity` is `min(capacity, total_weighted_demand)`, where `total_weighted_demand` is the sum of weighted floats. When `fill_capacity=False` and `capacity > total_weighted_demand`, `effective_capacity` is a float (e.g., 23.7). `gap = int(round(23.7 - 18)) = 6`, but the **integer** order total then becomes 24 > 23.7. In practice the response field `summary.total_orders` may exceed `total_weighted_demand` by ≤ 1. Minor, but it means "capacity is a cap, not a target" can be off by one due to rounding.

### Bug 12 — `api/main.py` silently drops items not present in the snapshot

```
df_slice = df_slice[df_slice["item_nbr"].isin(item_map.keys())]
```

If the caller passes 10 items and only 6 exist in the (store, date) snapshot rows, the API returns 6 results with no indication that 4 were dropped. The Streamlit UI's coverage stress test will then sum orders against `total_forecast` computed from only the surviving items, making the displayed "Demand Served" metric inconsistent across requests if the dropped set differs.

### Bug 13 — `scripts/build_test_snapshot_2016Q1.py` duplicates the date filter

The script applies the date mask both in its own `mask` expression and again inside `build_base_snapshot(... START_DATE, END_DATE)`. Not a bug, but redundant.

### Bug 14 — `scripts/build_featured_snapshot.py` and `build_test_featured_snapshot_2016Q1.py` duplicate the same feature block

Both scripts inline the same five feature-engineering steps with the same `lags`/`rolls` constants. `src/features/feature_pipeline.apply_all_features` exists for exactly this purpose but is unused. Risk: a future change to one builder (e.g., adding a new feature step) will silently diverge from the other unless the developer remembers both files, leading to train/serve skew.

### Bug 15 — Featured-snapshot builder for test does not call `sort_values` before lags

`build_featured_snapshot.py` sorts by `(store_nbr, item_nbr, date)` before calling `add_lag_features`. `build_test_featured_snapshot_2016Q1.py` also sorts (line 61), so this is fine in both — noting it because `add_lag_features` itself also sorts internally, so the outer sort is redundant.

### Bug 16 — `Dockerfile` does not include `ui/`

The image copies `api/`, `src/`, and `data/` but not `ui/`. The container therefore can serve the FastAPI backend only — the Streamlit UI cannot be launched from the published image. README/`run_app.ps1` document a two-process launch that assumes a local checkout, so this is consistent with that intent, but a reader expecting to `docker run` the full app will be surprised.

---

## 7. Open questions

These could not be resolved by static analysis alone.

1. **Is Bug 1 (no January history loaded for test lags) intentional?** The README's wording suggests it was meant to be fixed by loading January 2016 rows and then trimming after the lag step, but the code never loaded those rows. Treat this as a known degraded forecast for early-February rows, or fix the snapshot builder?
2. **Is the `data/snapshots/favorita_train_snapshot_2015/` directory (no `.parquet` suffix) leftover state or an alternative partitioned write?** Confirming requires inspecting its contents at runtime.
3. **Should `MODEL_VERSION_LABEL = "v1"` be updated to track `latest/metadata.json:version` automatically, or is it deliberately a manual label?** Two demo scripts depend on this constant pointing to an on-disk version directory.
4. **Were the deprecated scripts ever intended to run in their current state?** `scripts/deprecated/test_feature_pipeline.py` and `apply_sampling_universe.py` will raise `ImportError` if executed; if they are kept as "reference snapshots in time," that should be documented.
5. **What is the intended lifetime of `data/models/v1/`?** It is on disk but is not the active version. Promotion policy (when to delete old versions vs. keep for rollback) is not documented.
6. **Was a `tests/` directory ever planned?** None exists at the project root. The lack of automated tests, combined with two near-duplicate feature builders (§Bug 14), is a likely source of future train/serve skew.
7. **For the API, what is the expected behavior when a caller submits items the snapshot does not contain?** Currently dropped silently (§Bug 12). Should the response surface a `not_found` list?
8. **Is the integer over-allocation in `optimize_proportional_allocation` (§Bug 11) acceptable in the business domain?** A one-unit overshoot above stated capacity may or may not be tolerable depending on how the warehouse interprets the cap.
9. **Is `ACTIVE_DATASET_MODE` ever flipped to `"train"` in production, or only for local debugging?** The snapshot map supports both, but the README only references the test snapshot.
10. **Why does `run_app.ps1` launch the API without `--reload` but `Start-Sleep -Seconds 3` before Streamlit?** Three seconds may be insufficient for the API to finish loading the predictor and snapshot on cold start; Streamlit can come up before the API is ready, which the sidebar handles with an error stop — but it means the first launch sometimes shows "API not reachable" until refresh.
