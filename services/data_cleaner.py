from pathlib import Path
import pandas as pd

DATA_DIR = Path("data")

REQUIRED_COLUMNS = [
    "Date", "Ticker", "Open", "High", "Low", "Close", "Volume"
]


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Очистка одного OHLCV датасета:
    - приведение типов
    - удаление мусора
    - контроль контрактов
    """

    df = df.copy()

    # --- обязательные колонки ---
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # --- типы ---
    df["Date"] = pd.to_datetime(df["Date"], utc=True, errors="coerce")

    for col in ["Open", "High", "Low", "Close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce")

    # --- базовая фильтрация ---
    df = df.dropna(subset=["Date", "Ticker", "Open", "High", "Low", "Close"])

    # --- логические проверки OHLC ---
    df = df[
        (df["High"] >= df["Low"]) &
        (df["High"] >= df["Open"]) &
        (df["High"] >= df["Close"]) &
        (df["Low"] <= df["Open"]) &
        (df["Low"] <= df["Close"])
    ]

    # --- сортировка и дедупликация ---
    df = (
        df
        .drop_duplicates(subset=["Date", "Ticker"], keep="last")
        .sort_values(["Ticker", "Date"], ascending=[True, False])
        .reset_index(drop=True)
    )

    return df


def clean_all_datasets() -> None:
    """
    Очистка всех parquet-файлов в data/
    """

    parquet_files = list(DATA_DIR.glob("*.parquet"))

    if not parquet_files:
        print("No parquet files to clean.")
        return

    for file_path in parquet_files:
        try:
            df = pd.read_parquet(file_path)
            df_clean = clean_dataset(df)
            df_clean.to_parquet(file_path, index=False)

            print(f"Cleaned: {file_path.name} ({len(df)} → {len(df_clean)})")

        except Exception as e:
            print(f"Failed to clean {file_path.name}: {e}")
