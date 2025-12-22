import pandas as pd
from src.config import RAW_DIR

def main():
    train_path = RAW_DIR / "train.csv"

    print(f"Looking for train.csv at:\n{train_path}\n")

    df = pd.read_csv(train_path, nrows=5)

    print("✅ Successfully loaded train.csv")
    print("Shape (first 5 rows):", df.shape)
    print("Columns:")
    print(df.columns.tolist())

if __name__ == "__main__":
    main()
