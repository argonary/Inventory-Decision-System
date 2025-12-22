import lightgbm as lgb
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List


def train_lgbm_quantile(
    df: pd.DataFrame,
    features: List[str],
    target_col: str,
    quantile: float,
    model_path: Path,
    categorical_features: List[str],
) -> None:
    """
    Train and save a LightGBM quantile model.
    """

    X = df[features].copy()
    y = df[target_col]

    # Log transform target
    y_log = np.log1p(y.clip(lower=0))

    # Cast categoricals
    for col in categorical_features:
        X[col] = X[col].astype("category")

    dataset = lgb.Dataset(
        X,
        label=y_log,
        categorical_feature=categorical_features,
        free_raw_data=False,
    )

    params = {
        "objective": "quantile",
        "alpha": quantile,
        "metric": "quantile",
        "learning_rate": 0.05,
        "num_leaves": 64,
        "min_data_in_leaf": 100,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "verbosity": -1,
    }

    model = lgb.train(
        params=params,
        train_set=dataset,
        num_boost_round=300,
    )

    model.save_model(str(model_path))
