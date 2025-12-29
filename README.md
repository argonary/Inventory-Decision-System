# Inventory Decision System
*Turning demand uncertainty into capacity-constrained order recommendations*

## Overview

This project demonstrates how demand uncertainty can be translated into concrete, operational inventory decisions.

Instead of planning inventory using a single average forecast, the system plans
against high-demand scenarios and explicitly accounts for capacity constraints.
This reflects how real supply chains operate and allows decision-makers to choose
their risk posture intentionally.

The system is built on the Corporación Favorita grocery sales dataset and uses
quantile forecasting with LightGBM to produce feasible, capacity-aware order
quantities.

## Business Motivation

Inventory planning always involves tradeoffs.

  - Ordering too little leads to stockouts, lost sales, and poor customer experience.
  - Ordering too much increases holding costs, waste, and working capital requirements.

In practice, capacity is limited. Warehouses, suppliers, and transportation networks
cannot fulfill unlimited demand. Planning purely off average demand ignores both
uncertainty and these real operational limits.

This project shows how to:
  - Model demand uncertainty directly
  - Choose a clear and explicit risk posture
  - Convert forecasts into realistic order decisions

## What the System Does

At a high level, the system:

  - Forecasts daily demand for store-item combinations
  - Produces multiple demand scenarios using quantile models
  - Allocates limited capacity across SKUs
  - Generates order quantities that respect both demand and capacity

The output is an order plan that balances service level objectives with operational
feasibility.

## Key Concept: Quantile Forecasting (Plain English)

Traditional forecasting methods predict a single number, often interpreted as the
average expected demand.

Quantile forecasting predicts several demand scenarios instead.

For example:
  - Lower quantiles represent low-demand days
  - Middle quantiles represent typical demand
  - Higher quantiles represent high-demand days

Planning with a higher quantile means planning for a busier-than-average scenario.

In practical terms:
  - Higher quantiles reduce the risk of stockouts
  - They require carrying more inventory
  - The tradeoff between risk and cost becomes explicit

This makes the inventory decision a business choice rather than a hidden modeling
assumption.

## How Decisions Are Produced

The system follows a deterministic and transparent flow:

  1. Historical sales data is transformed into feature snapshots
  2. A LightGBM quantile model generates demand estimates
  3. A capacity allocation step distributes limited capacity across items
  4. Order quantities are capped so they never exceed forecasted demand
  5. Results are returned in a structured format for downstream use

Every step respects the chosen demand scenario and capacity constraint.

## Overall Architecture

  Streamlit UI (client)
    → FastAPI service
      → Quantile demand model
        → Capacity allocation logic
          → Order recommendations

The Streamlit application is a pure client. It sends requests to the API, displays
forecasts and order quantities, and visualizes the tradeoffs between capacity and
demand served.

## How to Run Locally

The system is split into two components:
  - A FastAPI backend that performs forecasting and order allocation
  - A Streamlit frontend that acts as a client and visualization layer

### Run the API (Docker)

  Build the Docker image:

      docker build -t favorita-api .

  Run the container:

      docker run -p 8000:8000 favorita-api

  Verify the service is running:

      http://127.0.0.1:8000/health

### Run the Streamlit UI

  In a separate terminal, run:

      streamlit run ui/app.py

  The UI will connect to the local API and display forecasts, order quantities,
  and capacity sensitivity curves.

## API Endpoints

The FastAPI service exposes the following endpoints:

    - POST /forecast-to-orders
      Accepts a payload describing SKUs, capacity, and planning scenario.
      Returns demand forecasts and recommended order quantities.

    - GET /health
      Simple health check endpoint.

    - GET /version
      Returns the current model and snapshot version.

## Repository Structure

    api/
        FastAPI service, request schemas, and inference logic

    ui/
        Streamlit application acting as a pure API client

    scripts/
        Snapshot building, feature engineering, and utility scripts

    data/
        snapshots/   Feature snapshots used for inference
        models/      Trained model artifacts (tracked with Git LFS)

## Data and Artifacts

This repository uses Git LFS to manage large artifacts such as:

    - Feature snapshots
    - Trained model files

Raw Corporación Favorita CSV files are intentionally excluded and remain local.
This keeps the repository lightweight while preserving reproducibility through
curated snapshots included in the repo.

## Limitations and Scope

This project is a demonstration system.

  - The UI operates on a curated snapshot for responsiveness
  - The dataset is historical and finite
  - Models are not retrained automatically

In a production setting, forecasts would be refreshed regularly, new data would
be ingested continuously, and capacity constraints could vary over time.

## Business Impact

This approach enables better operational decisions by:

  - Reducing stockout risk through explicit planning for high-demand scenarios
  - Preventing over-ordering beyond realistic demand
  - Making capacity constraints visible and actionable
  - Allowing stakeholders to reason clearly about risk versus cost

While this project is illustrative, the same framework can be extended to real
supply chain environments with minimal conceptual changes.

## Author

Aryan Pai
