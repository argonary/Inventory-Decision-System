"""Prefect orchestration of the rebuild pipeline.

Replicates the five sequential steps from rebuild_pipeline.ps1 as Prefect tasks
inside a single flow. Each script's main() is called in-process; PYTHONPATH is
set on os.environ so any dynamic imports inside the scripts still resolve.
"""
from __future__ import annotations

import os
import sys
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Ensure `scripts` and `src` are importable when running this file directly
# (python pipeline/prefect_pipeline.py adds pipeline/ to sys.path, not the root).
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from prefect import flow, task  # noqa: E402  (sys.path must be set first)


@task(
    name="Build base training snapshot 2013-2015",
    retries=1,
    retry_delay_seconds=30,
    log_prints=True,
)
def build_training_snapshot() -> None:
    from scripts import build_training_snapshot as mod
    mod.main()


@task(
    name="Build featured training snapshot",
    retries=1,
    retry_delay_seconds=30,
    log_prints=True,
)
def build_featured_snapshot() -> None:
    from scripts import build_featured_snapshot as mod
    mod.main()


@task(
    name="Build base test snapshot 2016 Q1",
    retries=1,
    retry_delay_seconds=30,
    log_prints=True,
)
def build_test_snapshot() -> None:
    from scripts import build_test_snapshot_2016Q1 as mod
    mod.main()


@task(
    name="Build featured test snapshot 2016 Q1",
    retries=1,
    retry_delay_seconds=30,
    log_prints=True,
)
def build_test_featured_snapshot() -> None:
    from scripts import build_test_featured_snapshot_2016Q1 as mod
    mod.main()


@task(
    name="Train quantile models and update latest",
    retries=1,
    retry_delay_seconds=30,
    log_prints=True,
)
def train_quantile_models(
    version: str,
    quantiles: list[float],
    update_latest: bool,
) -> None:
    from scripts import train_quantile_model as mod

    argv = [
        "train_quantile_model.py",
        "--version", version,
        "--quantiles", *[str(q) for q in quantiles],
    ]
    if update_latest:
        argv.append("--update-latest")

    original_argv = sys.argv
    try:
        sys.argv = argv
        mod.main()
    finally:
        sys.argv = original_argv


def _default_version() -> str:
    return f"v_{date.today().strftime('%Y_%m_%d')}"


@flow(name="rebuild_pipeline")
def rebuild_pipeline(
    version: str | None = None,
    quantiles: list[float] | None = None,
    update_latest: bool = True,
) -> None:
    # Set PYTHONPATH before any task runs so dynamic imports inside the scripts
    # (or any subprocesses they spawn) resolve src/ and scripts/ correctly.
    os.environ["PYTHONPATH"] = str(PROJECT_ROOT)

    if version is None:
        version = _default_version()
    if quantiles is None:
        quantiles = [0.90, 0.95]

    build_training_snapshot()
    build_featured_snapshot()
    build_test_snapshot()
    build_test_featured_snapshot()
    train_quantile_models(
        version=version,
        quantiles=quantiles,
        update_latest=update_latest,
    )


if __name__ == "__main__":
    rebuild_pipeline()
