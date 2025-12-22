import pandas as pd
from src.config import SNAPSHOTS_DIR
from src.ml.feature_config import FEATURES, TARGET_COL

df = pd.read_parquet(
    SNAPSHOTS_DIR / "favorita_train_model_table_2016Q1.parquet"
)

print("Missing feature columns:", set(FEATURES) - set(df.columns))
print("Missing target:", TARGET_COL not in df.columns)
