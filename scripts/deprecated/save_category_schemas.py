import pandas as pd
from src.config import SNAPSHOTS_DIR, MODELS_DIR
from src.ml.feature_config import CATEGORICAL_FEATURES
from src.features.categorical import (
    extract_category_schemas,
    save_category_schemas,
)

def main():
    df = pd.read_parquet(
        SNAPSHOTS_DIR / "favorita_train_model_table_2016Q1.parquet"
    )

    schemas = extract_category_schemas(df, CATEGORICAL_FEATURES)

    MODELS_DIR.mkdir(exist_ok=True)
    out_path = MODELS_DIR / "category_schemas.json"
    save_category_schemas(schemas, out_path)

    print(f"✅ Category schemas saved to {out_path}")

if __name__ == "__main__":
    main()
