from __future__ import annotations

from functools import lru_cache

import pandas as pd


DATA_PATH = "data/day_data.parquet"
BASE_COLUMNS = {"Date", "Ticker"}
PERCENT_SOURCE_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


def add_percentage_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in PERCENT_SOURCE_COLUMNS:
        if col in df.columns:
            df[f"{col}_perc"] = df.groupby("Ticker")[col].pct_change() * 100

    return df


@lru_cache(maxsize=1)
def load_day_df() -> pd.DataFrame:
    df = pd.read_parquet(DATA_PATH)
    if "Date" not in df.columns or "Ticker" not in df.columns:
        raise ValueError("day_data.parquet must contain Date and Ticker")

    df["Date"] = pd.to_datetime(df["Date"], utc=True, errors="coerce")
    return add_percentage_features(df)


def numeric_features(df: pd.DataFrame | None = None) -> list[str]:
    df = df if df is not None else load_day_df()
    return sorted(
        col
        for col in df.columns
        if col not in BASE_COLUMNS and pd.api.types.is_numeric_dtype(df[col])
    )


def tickers(df: pd.DataFrame | None = None) -> list[str]:
    df = df if df is not None else load_day_df()
    return sorted(df["Ticker"].dropna().unique().tolist())


def filter_ticker_range(
    df: pd.DataFrame,
    ticker: str,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    result = df[df["Ticker"] == ticker].copy()
    if result.empty:
        return result

    if start_date:
        start = pd.to_datetime(start_date, utc=True, errors="raise")
        result = result[result["Date"] >= start]

    if end_date:
        end = (
            pd.to_datetime(end_date, utc=True, errors="raise")
            + pd.Timedelta(days=1)
            - pd.Timedelta(microseconds=1)
        )
        result = result[result["Date"] <= end]

    return result.sort_values("Date")


def get_series(
    ticker: str,
    target: str,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    df = load_day_df()
    if target not in df.columns:
        raise ValueError(f"unknown target: {target}")

    result = filter_ticker_range(df, ticker, start_date, end_date)
    if result.empty:
        raise ValueError("no data for ticker/date range")

    result = result[["Date", target]].dropna().rename(columns={target: "value"})
    if result.empty:
        raise ValueError("no non-empty values for target")

    return result
