import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def make_features(series, lags, rolling_windows):
    df = pd.DataFrame({"y": series})

    for lag in lags:
        df[f"lag_{lag}"] = df["y"].shift(lag)

    for window in rolling_windows:
        shifted = df["y"].shift(1)
        rolling = shifted.rolling(window)
        df[f"rolling_mean_{window}"] = rolling.mean()
        df[f"rolling_std_{window}"] = rolling.std()
        df[f"rolling_min_{window}"] = rolling.min()
        df[f"rolling_max_{window}"] = rolling.max()

    return df.dropna()


def linear_forecast(
    series,
    horizon,
    model_type="ridge",
    lags=(1, 2, 3, 7, 14, 28),
    rolling_windows=(7, 14, 28),
    alpha=1.0,
    fit_intercept=True,
    max_iter=10000,
    scale_features=True,
):
    df = make_features(series, lags, rolling_windows)

    if df.empty:
        raise ValueError("Недостаточно данных для линейной модели")

    X = df.drop(columns=["y"])
    y = df["y"]

    if model_type == "linear":
        model = LinearRegression(fit_intercept=fit_intercept)
    elif model_type == "ridge":
        model = Ridge(alpha=alpha, fit_intercept=fit_intercept)
    elif model_type == "lasso":
        model = Lasso(alpha=alpha, fit_intercept=fit_intercept, max_iter=max_iter)
    else:
        raise ValueError("model_type должен быть: 'linear', 'ridge' или 'lasso'")

    steps = []
    if scale_features:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", model))

    pipeline = Pipeline(steps)
    pipeline.fit(X, y)

    history = list(series)
    preds = []
    feature_order = list(X.columns)

    for _ in range(horizon):
        features = {}

        for lag in lags:
            features[f"lag_{lag}"] = history[-lag] if lag <= len(history) else history[0]

        for window in rolling_windows:
            values = history[-window:] if len(history) >= window else history
            features[f"rolling_mean_{window}"] = np.mean(values)
            features[f"rolling_std_{window}"] = np.std(values)
            features[f"rolling_min_{window}"] = np.min(values)
            features[f"rolling_max_{window}"] = np.max(values)

        X_pred = pd.DataFrame(
            [[features[col] for col in feature_order]],
            columns=feature_order,
        )
        pred = pipeline.predict(X_pred)[0]
        preds.append(pred)
        history.append(pred)

    return np.array(preds), pipeline
