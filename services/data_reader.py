from pathlib import Path
import pandas as pd

PARQUET_FILES = {
    "1m": "minute_data.parquet",
    "15m": "15min_data.parquet",
    "1h": "hour_data.parquet",
    "1d": "day_data.parquet",
    "1w": "week_data.parquet",
}


def load_close_series(symbol: str, interval: str, limit: int = 300):
    if interval not in PARQUET_FILES:
        raise ValueError(f"Unknown interval: {interval}")

    file_path = Path("data") / PARQUET_FILES[interval]

    if not file_path.exists():
        raise FileNotFoundError(f"Parquet file not found: {file_path}")

    df = pd.read_parquet(file_path)

    time_col = "Datetime" if "Datetime" in df.columns else "Date"
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")

    df = df[df["Ticker"] == symbol]

    if df.empty:
        raise ValueError(f"No data for symbol {symbol}")

    df = df.sort_values(time_col).tail(limit)

    return df["Close"].astype(float).tolist()


def load_ohlcv_df(tickers, interval: str, start=None, end=None) -> pd.DataFrame:
    if interval not in PARQUET_FILES:
        raise ValueError(f"Unknown interval: {interval}")

    file_path = Path("data") / PARQUET_FILES[interval]

    if not file_path.exists():
        raise FileNotFoundError(f"Parquet file not found: {file_path}")

    df = pd.read_parquet(file_path)

    time_col = "Datetime" if "Datetime" in df.columns else "Date"
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")

    if tickers == "ALL":
        tickers = df["Ticker"].unique().tolist()
    elif isinstance(tickers, str):
        tickers = [tickers]

    df = df[df["Ticker"].isin(tickers)]

    if start is not None:
        df = df[df[time_col] >= pd.to_datetime(start)]
    if end is not None:
        df = df[df[time_col] <= pd.to_datetime(end)]

    df = df.sort_values(["Ticker", time_col])

    return df.reset_index(drop=True)