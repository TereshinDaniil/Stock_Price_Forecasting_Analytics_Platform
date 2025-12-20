import pandas as pd
from services.models.naive import (
    naive_forecast,
    seasonal_naive_forecast,
    moving_average_forecast,
    drift_forecast,
    exponential_smoothing_forecast,
)


def run_naive_model(
    ticker: str,
    target: str,
    horizon: int,
    model: str,
):
    # 1️⃣ загрузка дневных данных
    df = pd.read_parquet("data/day_data.parquet")
    df["Date"] = pd.to_datetime(df["Date"], utc=True)

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

    # 2️⃣ выбор модели
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

    else:
        raise ValueError("Неизвестная модель")

    # 3️⃣ даты прогноза
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
