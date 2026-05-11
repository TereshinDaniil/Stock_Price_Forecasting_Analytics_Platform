import pandas as pd
from services.models.naive import (
    naive_forecast,
    seasonal_naive_forecast,
    moving_average_forecast,
    drift_forecast,
    exponential_smoothing_forecast,
)
from services.models.linear import linear_forecast
from services.models.random_forest import random_forest_forecast
from services.data_service import get_series


def forecast_series(
    series,
    horizon: int,
    model: str,
    rnn_type: str = "lstm",
):
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

    elif model == "random_forest":
        forecast, _ = random_forest_forecast(series, horizon)

    elif model == "linear":
        forecast, _ = linear_forecast(series, horizon)

    elif model == "mlp":
        try:
            from services.models.mlp import mlp_forecast_torch
        except ImportError as exc:
            raise ValueError(
                "PyTorch не установлен. Установите torch или выберите другую модель."
            ) from exc

        forecast, _ = mlp_forecast_torch(series, horizon)

    elif model == "rnn":
        try:
            from services.models.rnn import rnn_forecast_torch
        except ImportError as exc:
            raise ValueError(
                "PyTorch не установлен. Установите torch или выберите другую модель."
            ) from exc

        forecast, _ = rnn_forecast_torch(series, horizon, rnn_type=rnn_type)

    elif model == "chronos":
        try:
            from services.models.chronos import chronos_forecast
        except ImportError as exc:
            raise ValueError(
                "Chronos не установлен. Установите chronos-forecasting или выберите другую модель."
            ) from exc

        forecast, _ = chronos_forecast(series, horizon)

    else:
        raise ValueError("Неизвестная модель")

    return forecast


def run_naive_model(
    ticker: str,
    target: str,
    horizon: int,
    model: str,
    rnn_type: str = "lstm",
):
    df = get_series(ticker=ticker, target=target)
    series = df["value"].dropna()
    forecast = forecast_series(series, horizon, model, rnn_type)

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
