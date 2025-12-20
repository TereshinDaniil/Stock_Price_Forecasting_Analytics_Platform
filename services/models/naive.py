import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing


def naive_forecast(series, horizon):
    last_value = series.iloc[-1]
    return np.repeat(last_value, horizon)


def seasonal_naive_forecast(series, horizon, season_length=7):
    values = series.iloc[-season_length:]
    repeats = horizon // season_length + 1
    return np.tile(values, repeats)[:horizon]


def moving_average_forecast(series, horizon, window=26, mode="shift"):
    history = list(series)
    forecasts = []

    for i in range(horizon):
        if mode == "shift":
            current_window = window
        elif mode == "expanding":
            current_window = window + i
        else:
            raise ValueError("mode должен быть 'shift' или 'expanding'")

        current_window = min(current_window, len(history))
        ma = np.mean(history[-current_window:])
        forecasts.append(ma)
        history.append(ma)

    return np.array(forecasts)


def drift_forecast(series, horizon, lookback=None):
    if lookback is not None:
        series = series.iloc[-lookback:]

    y1 = series.iloc[0]
    y_last = series.iloc[-1]
    n = len(series)

    drift = (y_last - y1) / (n - 1)
    return y_last + drift * np.arange(1, horizon + 1)


def exponential_smoothing_forecast(
    series,
    horizon,
    trend=None,
    damped=False,
    seasonal=None,
    seasonal_periods=None,
):
    model = ExponentialSmoothing(
        series,
        trend=trend,
        damped_trend=damped,
        seasonal=seasonal,
        seasonal_periods=seasonal_periods,
        initialization_method="estimated",
    )

    fitted = model.fit(optimized=True)
    forecast = fitted.forecast(horizon)

    return forecast
