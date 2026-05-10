import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


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


def random_forest_forecast(
    series,
    horizon,
    lags=(1, 2, 3, 7, 14, 28),
    rolling_windows=(7, 14, 28),
    n_estimators=200,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features="sqrt",
    bootstrap=True,
    random_state=42,
    n_jobs=-1,
):
    df = make_features(series, lags, rolling_windows)

    if df.empty:
        raise ValueError("Недостаточно данных для Random Forest")

    X = df.drop(columns=["y"])
    y = df["y"]

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        bootstrap=bootstrap,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    model.fit(X, y)

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
        pred = model.predict(X_pred)[0]
        preds.append(pred)
        history.append(pred)

    return np.array(preds), model
