from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlflow.pyfunc
import numpy as np
import pandas as pd


DEFAULT_LAGS = (1, 2, 3, 7, 14, 28)
DEFAULT_ROLLING_WINDOWS = (7, 14, 28)


@dataclass(frozen=True)
class ForecastModelConfig:
    ticker: str
    target: str
    model_name: str
    lags: tuple[int, ...] = DEFAULT_LAGS
    rolling_windows: tuple[int, ...] = DEFAULT_ROLLING_WINDOWS
    freq: str = "D"


def make_recursive_features(
    history: list[float],
    lags: tuple[int, ...],
    rolling_windows: tuple[int, ...],
    feature_order: list[str],
) -> pd.DataFrame:
    features: dict[str, float] = {}

    for lag in lags:
        features[f"lag_{lag}"] = history[-lag] if lag <= len(history) else history[0]

    for window in rolling_windows:
        values = history[-window:] if len(history) >= window else history
        features[f"rolling_mean_{window}"] = float(np.mean(values))
        features[f"rolling_std_{window}"] = float(np.std(values))
        features[f"rolling_min_{window}"] = float(np.min(values))
        features[f"rolling_max_{window}"] = float(np.max(values))

    return pd.DataFrame([[features[col] for col in feature_order]], columns=feature_order)


class RecursiveForecastPyFuncModel(mlflow.pyfunc.PythonModel):
    def __init__(
        self,
        config: ForecastModelConfig,
        history: list[float],
        last_date: str,
        estimator: Any | None = None,
        feature_order: list[str] | None = None,
    ):
        self.config = config
        self.history = [float(value) for value in history]
        self.last_date = last_date
        self.estimator = estimator
        self.feature_order = feature_order or []

    def _forecast(self, horizon: int, history_override: list[float] | None = None) -> np.ndarray:
        history = list(history_override) if history_override is not None else list(self.history)
        preds: list[float] = []

        for step in range(horizon):
            if self.config.model_name == "naive":
                pred = history[-1]
            elif self.config.model_name == "seasonal_naive":
                season_length = 7
                pred = history[-season_length + (step % season_length)]
            elif self.config.model_name == "moving_average":
                window = min(26, len(history))
                pred = float(np.mean(history[-window:]))
            elif self.config.model_name == "drift":
                drift = (history[-1] - history[0]) / max(len(history) - 1, 1)
                pred = float(history[-1] + drift)
            elif self.config.model_name in {"linear", "random_forest"}:
                if self.estimator is None or not self.feature_order:
                    raise ValueError("Для рекурсивных ML-моделей нужны estimator и порядок признаков.")
                X_pred = make_recursive_features(
                    history,
                    self.config.lags,
                    self.config.rolling_windows,
                    self.feature_order,
                )
                pred = float(self.estimator.predict(X_pred)[0])
            else:
                raise ValueError(f"Неподдерживаемая залоггированная модель: {self.config.model_name}")

            preds.append(float(pred))
            history.append(float(pred))

        return np.asarray(preds, dtype=float)

    def predict(self, context, model_input: pd.DataFrame) -> pd.DataFrame:
        if model_input is None or model_input.empty or "horizon" not in model_input:
            horizon = 30
        else:
            horizon = int(model_input["horizon"].iloc[0])

        history_override = None
        if model_input is not None and "history" in model_input and pd.notna(model_input["history"].iloc[0]):
            raw_history = model_input["history"].iloc[0]
            if isinstance(raw_history, str):
                history_override = [float(value) for value in raw_history.split(",") if value.strip()]
            else:
                history_override = [float(value) for value in raw_history]

        preds = self._forecast(horizon=horizon, history_override=history_override)
        dates = pd.date_range(
            start=pd.Timestamp(self.last_date) + pd.Timedelta(days=1),
            periods=horizon,
            freq=self.config.freq,
            tz="UTC",
        )

        return pd.DataFrame(
            {
                "date": dates.date.astype(str),
                "ticker": self.config.ticker,
                "target": self.config.target,
                "model": self.config.model_name,
                "prediction": preds,
            }
        )
