import numpy as np
import pandas as pd
import lightgbm as lgb


def make_features(series, lags, rolling_windows=(7, 14, 28)):
    df = pd.DataFrame({"y": series})

    for lag in lags:
        df[f"lag_{lag}"] = df["y"].shift(lag)

    for w in rolling_windows:
        df[f"rolling_mean_{w}"] = df["y"].shift(1).rolling(w).mean()
        df[f"rolling_std_{w}"] = df["y"].shift(1).rolling(w).std()
        df[f"rolling_min_{w}"] = df["y"].shift(1).rolling(w).min()
        df[f"rolling_max_{w}"] = df["y"].shift(1).rolling(w).max()

    df = df.dropna()
    return df


def lgb_forecast(
    series,
    horizon,
    lags=(1, 2, 3, 7, 14, 28),
    rolling_windows=(7, 14, 28),
    n_estimators=100,
    learning_rate=0.05,
    max_depth=-1
):
    df = make_features(series, lags, rolling_windows)

    X = df.drop(columns=["y"])
    y = df["y"]

    model = lgb.LGBMRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        n_jobs=-1
    )
    model.fit(X, y)

    history = list(series)
    preds = []

    feature_order = list(X.columns)

    for _ in range(horizon):
        features = {}

        for lag in lags:
            features[f"lag_{lag}"] = history[-lag] if lag <= len(history) else history[0]

        for w in rolling_windows:
            window = history[-w:] if len(history) >= w else history
            features[f"rolling_mean_{w}"] = np.mean(window)
            features[f"rolling_std_{w}"] = np.std(window)
            features[f"rolling_min_{w}"] = np.min(window)
            features[f"rolling_max_{w}"] = np.max(window)

        X_pred = pd.DataFrame([[features[col] for col in feature_order]], columns=feature_order)
        pred = model.predict(X_pred)[0]

        preds.append(pred)
        history.append(pred)

    return np.array(preds), model