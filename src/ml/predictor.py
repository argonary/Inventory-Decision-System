import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union

import numpy as np
import pandas as pd
import lightgbm as lgb

from src.ml.feature_config import FEATURES


@dataclass(frozen=True)
class ModelRegistry:
    """
    Maps service levels (quantiles) to model artifact paths.
    """
    models_by_alpha: Dict[float, Path]
    category_schema_path: Path


class QuantilePredictor:
    """
    Loads category schemas once, and loads one or more LightGBM quantile models.
    Provides a single interface to predict with a requested service level (alpha).
    """

    def __init__(self, registry: ModelRegistry):
        self.registry = registry

        # Load category schemas once
        with open(self.registry.category_schema_path, "r") as f:
            self.category_schemas: Dict[str, List[str]] = json.load(f)

        # Cache of loaded models
        self._models: Dict[float, lgb.Booster] = {}

    def _get_model(self, alpha: float) -> lgb.Booster:
        if alpha not in self.registry.models_by_alpha:
            raise ValueError(
                f"Unsupported service level {alpha}. "
                f"Supported: {sorted(self.registry.models_by_alpha.keys())}"
            )

        if alpha not in self._models:
            model_path = self.registry.models_by_alpha[alpha]
            self._models[alpha] = lgb.Booster(model_file=str(model_path))

        return self._models[alpha]

    def _apply_category_schemas(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Enforce training-time categorical schemas.
        Unseen categories become NaN (safe).
        """
        for col, categories in self.category_schemas.items():
            if col in df.columns:
                df[col] = pd.Categorical(df[col], categories=categories)
        return df

    def predict_df(
        self,
        df_features: pd.DataFrame,
        service_level: float = 0.90,
        clip_negative: bool = True,
    ) -> np.ndarray:
        """
        Predict quantiles for a DataFrame that already contains FEATURES.
        Returns predictions on original unit scale (inverts log1p).
        """
        df = df_features.copy()
        df = self._apply_category_schemas(df)

        X = df[FEATURES]
        model = self._get_model(service_level)

        y_hat_log = model.predict(X)
        y_hat = np.expm1(y_hat_log)

        if clip_negative:
            y_hat = np.clip(y_hat, a_min=0, a_max=None)

        return y_hat

    def predict_rows(
        self,
        rows: Union[List[dict], pd.DataFrame],
        service_level: float = 0.90,
    ) -> np.ndarray:
        """
        Convenience wrapper: accepts list-of-dicts or a DataFrame.
        Must contain FEATURES columns.
        """
        if isinstance(rows, list):
            df = pd.DataFrame(rows)
        else:
            df = rows

        return self.predict_df(df, service_level=service_level)
