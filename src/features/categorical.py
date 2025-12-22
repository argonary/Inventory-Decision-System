import json
import pandas as pd
from typing import Dict, List
from pathlib import Path


def extract_category_schemas(
    df: pd.DataFrame,
    categorical_features: List[str],
) -> Dict[str, List[str]]:
    schemas = {}

    for col in categorical_features:
        if col not in df.columns:
            raise ValueError(f"Missing categorical column: {col}")

        schemas[col] = (
            df[col]
            .astype("category")
            .cat
            .categories
            .tolist()
        )

    return schemas


def apply_category_schemas(
    df: pd.DataFrame,
    schemas: Dict[str, List[str]],
) -> pd.DataFrame:
    for col, categories in schemas.items():
        df[col] = pd.Categorical(df[col], categories=categories)
    return df


def save_category_schemas(
    schemas: Dict[str, List[str]],
    path: Path,
) -> None:
    with open(path, "w") as f:
        json.dump(schemas, f, indent=2)


def load_category_schemas(
    path: Path,
) -> Dict[str, List[str]]:
    with open(path) as f:
        return json.load(f)
