# Inventory Decision System

An end-to-end inventory ordering system built on the Corporacion Favorita grocery dataset. It uses LightGBM quantile regression to forecast demand at a chosen service level, then allocates order quantities across SKUs subject to a hard warehouse capacity constraint.

## Demo

[![Watch the demo](https://img.youtube.com/vi/bV7PRJ9Or-E/0.jpg)](https://youtu.be/bV7PRJ9Or-E)

## Architecture

The Streamlit frontend sends a decision request (store, date, SKUs, capacity, service level) to the FastAPI backend. The backend slices a pre-built feature snapshot, runs LightGBM quantile inference, and passes the forecasts to the proportional allocation optimizer, which returns integer order quantities that respect the capacity cap.

## Methodology

**Forecasting:** Two LightGBM models are trained on quantile loss, one at P90 and one at P95. Features include lag sales, rolling averages, calendar signals, oil price, holiday flags, store metadata, and promotion status. Targets are log1p-transformed at training time and inverted at inference.

**Optimization:** Given quantile forecasts across N SKUs and a capacity cap C, the optimizer runs proportional allocation with largest-remainder rounding to produce integer order quantities. Capacity is treated as a maximum, not a target. If total forecast is below capacity, orders match forecast exactly. An optional service floor ratio guarantees a minimum allocation fraction per SKU.

**Out-of-time evaluation:** The API serves a 2016Q1 feature snapshot, which is outside the 2013-2015 training window, giving a realistic demonstration of model generalization.

## Quickstart

Requirements: Python 3.11, Git LFS

Clone the repo and install dependencies:

    git clone https://github.com/argonary/Inventory-Decision-System.git
    cd Inventory-Decision-System
    python -m venv .venv
    .venv\Scripts\Activate.ps1
    pip install -r requirements.txt

Download the Corporacion Favorita dataset from Kaggle and place the CSVs in data/raw/, then build the featured snapshots:

    python scripts/build_training_snapshot.py
    python scripts/build_featured_snapshot.py
    python scripts/build_test_snapshot_2016Q1.py
    python scripts/build_test_featured_snapshot_2016Q1.py

Launch the full application:

    .\run_app.ps1

This opens the FastAPI backend and Streamlit frontend in separate terminals. The browser will open automatically at http://localhost:8501.

## API

The FastAPI backend exposes three endpoints:

- GET /health -- liveness check
- GET /version -- active model version and snapshot
- POST /forecast-to-orders -- main inference endpoint

Interactive API docs are available at http://localhost:8000/docs when the server is running.

## Dataset

Corporacion Favorita Grocery Sales Forecasting (Kaggle). The raw data is not included in this repo. Download it from Kaggle and place the CSVs in data/raw/.

## Tech Stack

- LightGBM 4.6 -- quantile regression
- FastAPI 0.126 -- REST backend
- Streamlit 1.52 -- interactive frontend
- pandas, numpy, pyarrow -- data and feature engineering
- Plotly -- capacity curve visualization
- Git LFS -- model artifact storage
