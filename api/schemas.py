from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


ForecastModel = Literal[
    "naive",
    "seasonal_naive",
    "moving_average",
    "drift",
    "exp_smoothing",
    "random_forest",
    "linear",
    "mlp",
    "rnn",
    "chronos",
]


class Point(BaseModel):
    date: str
    value: float


class XYPoint(BaseModel):
    x: float
    y: float


class SeriesResponse(BaseModel):
    status: str = "ok"
    ticker: str
    target: str
    kind: Literal["value", "returns"]
    points: list[Point]


class CorrelationResponse(BaseModel):
    status: str = "ok"
    ticker: str
    features: list[str]
    matrix: list[list[float]]


class AcfPacfResponse(BaseModel):
    status: str = "ok"
    ticker: str
    feature: str
    lags: list[int]
    acf: list[float]
    pacf: list[float]


class LagPlotResponse(BaseModel):
    status: str = "ok"
    ticker: str
    feature: str
    lag: int
    points: list[XYPoint]


class StructuralBreaksResponse(BaseModel):
    status: str = "ok"
    ticker: str
    target: str = "Close"
    points: list[Point]
    kernel_breaks: list[str]
    pelt_breaks: list[str]
    kernel_n_bkps: int
    pelt_penalty: int


class ForecastRequest(BaseModel):
    ticker: str
    target: str = "Close"
    horizon: int = Field(..., ge=1, le=365)
    model: ForecastModel = "naive"
    rnn_type: Literal["lstm", "gru"] = "lstm"


class ForecastResponse(BaseModel):
    status: str = "ok"
    ticker: str
    target: str
    model: ForecastModel
    rnn_type: Literal["lstm", "gru"] | None = None
    horizon: int
    history_window: int
    history: list[Point]
    forecast: list[Point]


class EvaluationRequest(BaseModel):
    ticker: str
    target: str = "Close"
    model: ForecastModel = "naive"
    rnn_type: Literal["lstm", "gru"] = "lstm"
    test_size: int = Field(30, ge=1, le=365)


class Metrics(BaseModel):
    mae: float
    rmse: float
    mape: float | None = None


class EvaluationResponse(BaseModel):
    status: str = "ok"
    ticker: str
    target: str
    model: ForecastModel
    rnn_type: Literal["lstm", "gru"] | None = None
    train_size: int
    test_size: int
    metrics: Metrics
    train: list[Point]
    test: list[Point]
    prediction: list[Point]
