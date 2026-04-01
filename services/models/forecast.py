import pandas as pd
from services.models.naive import (
    naive_forecast,
    seasonal_naive_forecast,
    moving_average_forecast,
    drift_forecast,
    exponential_smoothing_forecast,
)
from services.models.lgb import lgb_forecast


def add_percentage_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    cols = ["Open", "High", "Low", "Close", "Volume"]

    for col in cols:
        if col in df.columns:
            df[f"{col}_perc"] = (
                df.groupby("Ticker")[col]
                .pct_change() * 100
            )

    return df


def run_naive_model(
    ticker: str,
    target: str,
    horizon: int,
    model: str,
):
    df = pd.read_parquet("data/day_data.parquet")
    df["Date"] = pd.to_datetime(df["Date"], utc=True, errors="coerce")
    df = add_percentage_features(df)

    df = (
        df[df["Ticker"] == ticker]
        .sort_values("Date")
        .reset_index(drop=True)
    )

    if df.empty:
        raise ValueError("Нет данных для выбранного тикера")

    if target not in df.columns:
        raise ValueError(f"Признак '{target}' не найден")

    series = df[target].dropna()

    if len(series) < 30:
        raise ValueError("Недостаточно данных для прогноза")

    if model == "naive":
        forecast = naive_forecast(series, horizon)

    elif model == "seasonal_naive":
        forecast = seasonal_naive_forecast(series, horizon)

    elif model == "moving_average":
        forecast = moving_average_forecast(series, horizon)

    elif model == "drift":
        forecast = drift_forecast(series, horizon)

    elif model == "exp_smoothing":
        forecast = exponential_smoothing_forecast(series, horizon)

    elif model == "lgb":
        forecast, _ = lgb_forecast(series, horizon)

    else:
        raise ValueError("Неизвестная модель")

    last_date = df["Date"].iloc[-1]
    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        periods=horizon,
        freq="D",
        tz="UTC",
    )

    return pd.DataFrame({
        "Date": future_dates,
        "Forecast": forecast
    })