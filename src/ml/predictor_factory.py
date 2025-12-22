from pathlib import Path
from typing import Optional

from src.config import MODELS_DIR
from src.ml.predictor import ModelRegistry, QuantilePredictor


def build_predictor(
    version: Optional[str] = None,
) -> QuantilePredictor:
    """
    Build a QuantilePredictor for a specific model version.

    Args:
        version:
            - None: use MODELS_DIR / "latest"
            - str:  use MODELS_DIR / <version>
    """

    if version is None:
        model_dir = MODELS_DIR / "latest"
    else:
        model_dir = MODELS_DIR / version

    if not model_dir.exists():
        raise FileNotFoundError(
            f"Model directory not found: {model_dir}"
        )

    registry = ModelRegistry(
        models_by_alpha={
            0.90: model_dir / "favorita_lgbm_p90.txt",
            0.95: model_dir / "favorita_lgbm_p95.txt",
        },
        category_schema_path=model_dir / "category_schemas.json",
    )

    return QuantilePredictor(registry=registry)


def build_default_predictor() -> QuantilePredictor:
    """
    Backward-compatible default predictor.
    Uses MODELS_DIR / 'latest'.
    """
    return build_predictor(version=None)
