import pandas as pd
from pathlib import Path
import pandas_market_calendars as mcal

DATA_DIR = Path("data")

FILES = {
    "1m": "minute_data.parquet",
    "15min": "15min_data.parquet",
    "1h": "hour_data.parquet",
    "1d": "day_data.parquet",
    "1w": "week_data.parquet",
}

INTRADAY_STEP = {
    "1m": pd.Timedelta(minutes=1),
    "15min": pd.Timedelta(minutes=15),
    "1h": pd.Timedelta(hours=1),
}

MAX_SESSION_GAP = pd.Timedelta(hours=20)  # ночь + запас

nyse = mcal.get_calendar("NYSE")


def check_intraday(df: pd.DataFrame, interval: str):
    print("head():")
    print(df.head())

    dupes = df.duplicated(subset=["Ticker", "Date"]).sum()
    print("duplicates:", dupes)

    step = INTRADAY_STEP[interval]

    for ticker, g in df.groupby("Ticker"):
        g = g.sort_values("Date")
        diffs = g["Date"].diff().dropna()

        # считаем дырой только аномально большие разрывы
        bad = diffs[diffs > MAX_SESSION_GAP]

        if not bad.empty:
            print(f"❌ {ticker} abnormal gaps:")
            print(bad.head())
        else:
            print(f"✅ {ticker} — OK")


def check_daily(df: pd.DataFrame):
    print("head():")
    print(df.head())

    dupes = df.duplicated(subset=["Ticker", "Date"]).sum()
    print("duplicates:", dupes)

    cutoff = pd.Timestamp.utcnow() - pd.Timedelta(days=60)
    df_recent = df[df["Date"] >= cutoff]

    for ticker, g in df_recent.groupby("Ticker"):
        dates = g["Date"].dt.normalize().sort_values()

        schedule = nyse.schedule(
            start_date=dates.min().date(),
            end_date=dates.max().date()
        )
        trading_days = pd.to_datetime(schedule.index).tz_localize("UTC")

        missing = trading_days.difference(dates)

        if len(missing) > 0:
            print(f"❌ {ticker} missing trading days:", list(missing[:5]))
        else:
            print(f"✅ {ticker} — OK")


def check_weekly(df: pd.DataFrame):
    print("head():")
    print(df.head())

    dupes = df.duplicated(subset=["Ticker", "Date"]).sum()
    print("duplicates:", dupes)

    for ticker, g in df.groupby("Ticker"):
        g = g.sort_values("Date")
        diffs = g["Date"].diff().dropna()

        # ожидаем шаг ~7 дней
        bad = diffs[(diffs < pd.Timedelta(days=6)) | (diffs > pd.Timedelta(days=8))]

        if not bad.empty:
            print(f"❌ {ticker} irregular weekly steps:")
            print(bad.head())
        else:
            print(f"✅ {ticker} — OK")


def main():
    for interval, filename in FILES.items():
        print("\n" + "=" * 50)
        print(f"CHECK {interval} ({filename})")
        print("=" * 50)

        path = DATA_DIR / filename
        if not path.exists():
            print("❌ file not found")
            continue

        df = pd.read_parquet(path)

        if interval in ["1m", "15min", "1h"]:
            check_intraday(df, interval)
        elif interval == "1d":
            check_daily(df)
        elif interval == "1w":
            check_weekly(df)


if __name__ == "__main__":
    main()
