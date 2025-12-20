import time
from datetime import datetime, timedelta
from pathlib import Path
from services.data_cleaner import clean_all_datasets

import pandas as pd
from twelvedata import TDClient
from dotenv import load_dotenv
import os

# =====================
# CONFIG
# =====================

load_dotenv()
API_KEY = os.getenv("API12DATA")

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

TICKERS = [
    "AAPL", "NVDA", "TSLA", "MSFT", "AMZN",
    "INTC", "COST", "META", "AMD", "MCD"
]

PARQUET_FILES = {
    "1m": "minute_data.parquet",
    "15min": "15min_data.parquet",
    "1h": "hour_data.parquet",
    "1d": "day_data.parquet",
    "1w": "week_data.parquet",
}


TIMEFRAME_CONFIG = {
    "1m":    {"td_interval": "1min",   "step_days": 12},
    "15min": {"td_interval": "15min",  "step_days": 60},
    "1h":    {"td_interval": "1h",     "step_days": 182},
    "1d":    {"td_interval": "1day",   "step_days": 1825},
    "1w":    {"td_interval": "1week",  "step_days": 3650},
}


REQUIRED_COLUMNS = [
    "Date", "Ticker", "Open", "High", "Low", "Close", "Volume"
]

td = TDClient(apikey=API_KEY)

# =====================
# FETCH (STAGING)
# =====================

def fetch_twelvedata(symbol, interval, start, end) -> pd.DataFrame:
    df = td.time_series(
        symbol=symbol,
        interval=interval,
        start_date=start.strftime("%Y-%m-%d %H:%M:%S"),
        end_date=end.strftime("%Y-%m-%d %H:%M:%S"),
        outputsize=5000
    ).as_pandas()

    if df is None or df.empty:
        return pd.DataFrame()

    return df.reset_index()  # datetime остаётся как есть


# =====================
# NORMALIZE (CORE)
# =====================

def normalize_ohlcv(staging_df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    df = staging_df.rename(columns={
        "datetime": "Date",
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume"
    }).copy()

    df["Ticker"] = ticker
    df["Date"] = pd.to_datetime(df["Date"], utc=True, errors="coerce")

    df = df[REQUIRED_COLUMNS]

    return df


# =====================
# UPDATE ONE INTERVAL
# =====================

def update_table(interval: str):
    cfg = TIMEFRAME_CONFIG[interval]
    file_path = DATA_DIR / PARQUET_FILES[interval]

    # --- load existing parquet ---
    if file_path.exists():
        df = pd.read_parquet(file_path)

        # миграция старых файлов
        if "Date" not in df.columns and "Datetime" in df.columns:
            df = df.rename(columns={"Datetime": "Date"})

        df["Date"] = pd.to_datetime(df["Date"], utc=True, errors="coerce")
    else:
        df = pd.DataFrame(columns=REQUIRED_COLUMNS)

    now = pd.Timestamp.utcnow()
    new_parts = []

    for ticker in TICKERS:
        last_date = df.loc[df["Ticker"] == ticker, "Date"].max()

        if pd.isna(last_date):
            start = now - timedelta(days=cfg["step_days"])
        else:
            start = last_date + timedelta(minutes=1)

        step = timedelta(days=cfg["step_days"])

        while start < now:
            end = min(start + step, now)

            try:
                raw = fetch_twelvedata(
                    symbol=ticker,
                    interval=cfg["td_interval"],
                    start=start,
                    end=end
                )

                if not raw.empty:
                    normalized = normalize_ohlcv(raw, ticker)
                    new_parts.append(normalized)

            except Exception as e:
                print(f"Error {ticker} ({interval}): {e}")
                time.sleep(2)

            time.sleep(7)
            start = end

    if new_parts:
        new_data = pd.concat(new_parts, ignore_index=True)

        df_updated = (
            pd.concat([df, new_data], ignore_index=True)
              .drop_duplicates(subset=["Date", "Ticker"], keep="last")
              .sort_values(["Ticker", "Date"], ascending=[True, False])
              .reset_index(drop=True)
        )

        df_updated.to_parquet(file_path, index=False)
        print(f"[{interval}] added {len(new_data)} rows")
    else:
        print(f"[{interval}] no new data")


# =====================
# FULL PIPELINE
# =====================

def update_all():
    print("Starting data update...\n")

    for interval in PARQUET_FILES:
        update_table(interval)

    print("\nCleaning datasets...\n")
    clean_all_datasets()

    print("\nUpdate & cleaning finished.")



if __name__ == "__main__":
    update_all()
