from typing import List, Set
import pandas as pd


def check_required_columns(
    df: pd.DataFrame,
    features: List[str],
    target: str,
) -> Set[str]:
    """
    Return missing required columns.
    """
    required = set(features) | {target}
    return required - set(df.columns)
