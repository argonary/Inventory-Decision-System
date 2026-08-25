# Inventory Decision System

A daily order-recommendation system for grocery retail. Given a store, a date,
a list of SKUs, and a warehouse capacity, the system forecasts demand at a
chosen service level (P90 or P95) using LightGBM quantile regression, then runs
a capacity-constrained optimizer to produce integer order quantities per SKU.
The end-to-end stack covers raw-data ingestion, feature engineering, model
training, a FastAPI inference service, and a Streamlit decision UI, all of which is 
orchestrated by Prefect and deployable with Docker Compose.

## Demo

[![Watch the demo](https://img.youtube.com/vi/bV7PRJ9Or-E/0.jpg)](https://youtu.be/bV7PRJ9Or-E)

## Business Motivation

Grocery retailers face a daily ordering decision for every SKU in every store:
how many units to order given uncertain demand, finite warehouse capacity, and
the spoilage risk attached to perishable categories. The stakes on either side
are not symmetric.

- **Understocking** loses sales, frustrates customers, and erodes basket size
  in adjacent categories.
- **Overstocking** ties up working capital, fills shelf space that could earn
  more elsewhere, and — for perishables — turns directly into spoilage and
  markdown costs.

A point forecast that predicts "expected demand" gives you no lever for that
tradeoff. Quantile regression does. Instead of estimating the mean, a
quantile model estimates a high-confidence upper bound: the P90 forecast
is the level of demand the SKU will not exceed on 90% of days. That maps
directly to a business-level service target — "stock enough to serve demand
90% of the time" — and lets the operator dial the service level up or down
per run without retraining.

On top of the quantile forecast, the system runs an optimizer that allocates
order quantities across the SKUs in the request, respecting a hard warehouse
capacity ceiling and optionally up-weighting perishable items so coverage is
preserved where spoilage risk is highest. The output is decision-ready: a
per-SKU integer order quantity that fits the operational constraint, not
just a raw forecast.

## Architecture

```
                        ┌─────────────────────────────────────────┐
                        │  Prefect flow (rebuild_pipeline)        │
                        │  ───────────────────────────────────    │
   data/raw/*.csv  ──▶  │   1. base training snapshot 2013-2015   │
                        │   2. featured training snapshot         │
                        │   3. base test snapshot 2016 Q1         │
                        │   4. featured test snapshot 2016 Q1     │
                        │   5. train quantile models → latest/    │
                        └────────────────────┬────────────────────┘
                                             │
                                             ▼
                                  data/models/latest/
                                  data/snapshots/*.parquet
                                             │
                                             ▼
                                ┌────────────────────────┐
                                │  FastAPI  (port 8000)  │
                                │  /health  /version     │
                                │  /forecast-to-orders   │
                                └───────────┬────────────┘
                                            │
                                            ▼
                                ┌────────────────────────┐
                                │  Streamlit  (port 8501)│
                                │  decision UI           │
                                └────────────────────────┘

   Deployment layer: Docker Compose (api + ui services)
   Orchestration layer: Prefect (rebuild flow)
```

## Key Technical Decisions

- **LightGBM quantile regression, not point forecasts.** A point forecast forces
  the operator to pick an arbitrary safety stock multiplier downstream. A
  quantile model bakes the service-level decision directly into the prediction:
  the P90 head returns the demand level we expect to meet 90% of the time.
  LightGBM was chosen because it natively supports the pinball loss and trains
  quickly on the tabular lag/calendar/promotion feature set.

- **Two optimizers (proportional vs LP) chosen per request.** The proportional
  allocator with largest-remainder rounding is cheap, deterministic, and gives
  intuitive "everyone gets a fair share of capacity" behavior. The LP backend
  (`scipy.optimize.linprog`, HiGHS) is preferable when service floors and
  perishable weights interact non-trivially and a globally optimal allocation
  is worth the extra cost. Both share the same signature so callers swap via
  the `optimizer` field on the request.

- **Parquet snapshots, not live CSV queries.** The raw Favorita dataset is
  ~5 GB across multiple CSVs; rebuilding lag features per request would make
  the API unusable. Snapshots pre-compute features once per training/test
  window and are served via `pyarrow` for fast columnar slicing.

- **Versioned model artifacts with a `latest/` pointer.** Every training run
  writes to `data/models/v_YYYY_MM_DD/` with its own metadata and category
  schemas, and the training script updates `data/models/latest/` to point at
  the new version. This keeps prior models reproducible and makes rollback a
  one-line change without touching API code.

- **Prefect for orchestration, not a plain script.** The rebuild has five
  sequential steps and any one of them can fail on bad raw data. Prefect gives
  retries, structured logs per task, and a UI for inspecting failures, which
  capabilities a shell script can only approximate. `rebuild_pipeline.ps1` is
  kept around as a lightweight alternative for users who don't want Prefect.

- **Docker Compose for deployment.** The api and ui are different runtimes
  (FastAPI + LightGBM vs Streamlit + Plotly) with different dependency
  surfaces. Compose isolates them into two images, wires the UI to the API
  via a healthcheck-gated `depends_on`, and avoids the "works on my machine"
  drift from local virtualenvs.

## Quickstart

### A) Docker Compose

```
docker compose up --build
```

Then open <http://localhost:8501>. The `ui` service waits for the API's
`/health` check to pass before starting Streamlit.

### B) Local (PowerShell)

Requirements: Python 3.11, Git LFS. The processed snapshots and trained
models are tracked in Git LFS and pulled on clone, so you can skip the
pipeline rebuild if you only want to run the app.

```
git lfs install
git clone https://github.com/argonary/Inventory-Decision-System.git
cd Inventory-Decision-System
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Rebuild the pipeline (Prefect-orchestrated; requires raw CSVs in `data/raw/`):

```
.\run_pipeline.ps1
```

Launch the API and UI in two terminals:

```
.\run_app.ps1
```

The browser opens automatically at <http://localhost:8501>.

## Pipeline Steps

The Prefect flow in `pipeline/prefect_pipeline.py` runs these five tasks
sequentially:

1. **Build base training snapshot 2013-2015:** joins raw sales, items,
   stores, holidays, and oil CSVs into a single parquet table covering the
   training window.
2. **Build featured training snapshot:** applies the feature pipeline
   (calendar, holidays, oil, promotion, lag, categorical) to the base
   training snapshot.
3. **Build base test snapshot 2016 Q1:** assembles the out-of-time test
   table from raw CSVs, with January 2016 kept available so 28-day lags are
   fully populated when the test window starts in February.
4. **Build featured test snapshot 2016 Q1:** applies the same feature
   pipeline to the test snapshot so the API can serve it directly.
5. **Train quantile models and update latest:** trains a LightGBM quantile
   model per requested quantile (default 0.90, 0.95), writes versioned
   artifacts to `data/models/v_<date>/`, and updates `data/models/latest/`.

## API Reference

### `GET /health`

Liveness probe. Returns:

```json
{ "status": "ok" }
```

### `GET /version`

Returns the active model and snapshot identifiers:

```json
{
  "model_version": "v_2026_05_23",
  "dataset_mode": "test",
  "snapshot": "favorita_test_featured_2016Q1.parquet"
}
```

### `POST /forecast-to-orders`

Main inference endpoint. Validates the request, slices the featured
snapshot to the requested store/date/SKUs, runs quantile inference, and
returns the optimized allocation.

**Request body**

| Field                  | Type                  | Required | Default          | Notes                                                       |
|------------------------|-----------------------|----------|------------------|-------------------------------------------------------------|
| `date`                 | string (YYYY-MM-DD)   | yes      | —                | Decision date.                                              |
| `store_nbr`            | int                   | yes      | —                | Store number.                                               |
| `service_level`        | `"p90"` or `"p95"`    | yes      | —                | Case-insensitive. Selects the quantile model head.          |
| `items`                | list of `BatchItem`   | yes      | —                | Each item: `{ "item_nbr": int, "onpromotion": bool }`.      |
| `capacity_units`       | int > 0               | yes      | —                | Hard capacity ceiling (units, not currency).                |
| `service_floor_ratio`  | float in `[0.0, 1.0]` | no       | `0.0`            | Minimum fraction of forecast guaranteed per SKU.            |
| `perishable_weight`    | float > 0             | no       | `1.0`            | Multiplier applied to perishable items in the optimizer.    |
| `optimizer`            | `"proportional"` or `"lp"` | no   | `"proportional"` | Allocator backend.                                          |

**Response body**

| Field            | Type                  | Notes                                                       |
|------------------|-----------------------|-------------------------------------------------------------|
| `store_nbr`      | int                   | Echo of the request.                                        |
| `date`           | string                | Echo of the request.                                        |
| `service_level`  | string                | Echo of the request (normalized to lowercase).              |
| `capacity_units` | int                   | Echo of the request.                                        |
| `fill_capacity`  | bool                  | Whether the optimizer treated capacity as exact (currently always `false`). |
| `model_version`  | string                | From `data/models/latest/metadata.json`.                    |
| `dataset_mode`   | string                | `"train"` or `"test"`; configured via `src/config.py`.      |
| `snapshot`       | string                | Filename of the featured snapshot the API has loaded.       |
| `summary`        | object                | `{ "total_forecast": float, "total_orders": int }`.         |
| `results`        | list of object        | `{ "item_nbr": int, "forecast": float, "order_qty": int }`. |
| `not_found`      | list of int           | `item_nbr` values requested but missing from the snapshot for the given store/date. |

Interactive docs at <http://localhost:8000/docs> when the API is running.

## Model Details

From `data/models/latest/metadata.json`:

- **Version:** `v_2026_05_23`
- **Trained at:** `2026-05-23T20:12:33Z`
- **Quantiles trained:** `0.90`, `0.95`
- **Training window:** `2013-01-01` → `2015-06-30`
- **Validation window:** `2015-07-01` → `2015-12-31`
- **Training dataset:** `favorita_train_featured_2015.parquet`

Evaluation metrics are not recorded in `metadata.json`. To compute
out-of-time calibration and coverage metrics against the 2016 Q1 test
snapshot, run:

```
python scripts/evaluate_quantile_calibration_2016Q1.py
```

## Project Structure

```
Inventory-Decision-System/
├── api/                              # FastAPI inference service
│   ├── main.py                       # Endpoints + predictor + snapshot bootstrap
│   └── schemas.py                    # Pydantic request/response models
├── ui/                               # Streamlit decision UI
│   ├── app.py                        # Interactive client (reads API_BASE_URL env)
│   └── test_payload.csv              # Demo SKU payload for "Load demo payload"
├── src/                              # Library code shared by api/, scripts/, pipeline/
│   ├── config.py                     # Paths, model version label, dataset mode
│   ├── logging_config.py             # Structured-logging setup
│   ├── data/
│   │   ├── sampling.py               # Top-stores + top-items universe selection
│   │   ├── snapshot_builder.py       # Raw-CSV → parquet base snapshot
│   │   └── validation.py
│   ├── features/                     # Feature engineering modules
│   │   ├── calendar.py               # Day-of-week, month, payday-window flags
│   │   ├── categorical.py            # Stable category code mapping
│   │   ├── feature_pipeline.py       # apply_all_features() orchestrator
│   │   ├── holidays.py
│   │   ├── lags.py                   # 7/14/28-day lag and rolling features
│   │   ├── oil.py
│   │   └── promotion.py
│   ├── ml/
│   │   ├── feature_config.py         # Canonical feature list + dtypes
│   │   ├── predictor.py              # Quantile-aware inference wrapper
│   │   ├── predictor_factory.py      # Builds default predictor from latest/
│   │   ├── splits.py                 # Train/valid window splits
│   │   └── trainer.py                # LightGBM quantile training loop
│   ├── optimization/
│   │   └── optimizer.py              # Proportional + LP allocators
│   └── validation/
│       ├── feature_validation.py
│       └── schema_checks.py
├── scripts/                          # CLI entrypoints
│   ├── build_training_snapshot.py            # Pipeline step 1
│   ├── build_featured_snapshot.py            # Pipeline step 2
│   ├── build_test_snapshot_2016Q1.py         # Pipeline step 3
│   ├── build_test_featured_snapshot_2016Q1.py# Pipeline step 4
│   ├── train_quantile_model.py               # Pipeline step 5
│   ├── evaluate_quantile_calibration_2016Q1.py  # OOT calibration metrics
│   ├── validate_quantile_model.py            # Sanity checks on a trained version
│   ├── debug_pick_valid_request.py
│   └── dump_repo_tree.py
├── pipeline/
│   └── prefect_pipeline.py           # Prefect flow wrapping the five pipeline steps
├── tests/
│   ├── conftest.py
│   ├── test_optimizer.py             # Optimizer property + edge-case tests
│   └── test_schemas.py               # API schema validation tests
├── data/
│   └── models/
│       ├── latest/                   # Active model pointer (read by the API)
│       └── v_YYYY_MM_DD/             # Versioned artifacts written by training
├── notebooks/
│   └── data_preparation.ipynb        # Exploratory data-prep walkthrough
├── docs/
├── Dockerfile                        # API image (Python 3.11 + LightGBM runtime)
├── docker-compose.yml                # api + ui stack
├── .dockerignore
├── requirements.txt                  # Pinned runtime deps
├── requirements_raw.txt              # Unpinned source-of-truth deps
├── pytest.ini
├── rebuild_pipeline.ps1              # Sequential shell rebuild (no Prefect)
├── run_pipeline.ps1                  # Prefect-orchestrated rebuild
└── run_app.ps1                       # Launches API + UI in two terminals
```

`data/raw/`, `data/snapshots/`, `.venv/`, and `__pycache__/` directories
exist on disk but are intentionally omitted from this view.

## Acknowledgements

The dataset is from the
[Corporacion Favorita Grocery Sales Forecasting](https://www.kaggle.com/competitions/favorita-grocery-sales-forecasting)
Kaggle competition.
