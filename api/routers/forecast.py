from __future__ import annotations

import numpy as np
import pandas as pd
from fastapi import APIRouter, Body, HTTPException

from api.schemas import (
    EvaluationRequest,
    EvaluationResponse,
    ForecastRequest,
    ForecastResponse,
    Metrics,
    Point,
)
from services.data_service import get_series
from services.models.forecast import forecast_series, run_naive_model


router = APIRouter(tags=["Forecast"])


def _forecast_response(request: ForecastRequest) -> ForecastResponse:
    forecast_df = run_naive_model(
        ticker=request.ticker,
        target=request.target,
        horizon=request.horizon,
        model=request.model,
        rnn_type=request.rnn_type,
    )

    history_window = int(request.horizon) * 5
    history = get_series(request.ticker, request.target).tail(history_window)

    return ForecastResponse(
        ticker=request.ticker,
        target=request.target,
        model=request.model,
        rnn_type=request.rnn_type if request.model == "rnn" else None,
        horizon=request.horizon,
        history_window=history_window,
        history=[
            Point(date=row["Date"].date().isoformat(), value=float(row["value"]))
            for _, row in history.iterrows()
        ],
        forecast=[
            Point(date=row["Date"].date().isoformat(), value=float(row["Forecast"]))
            for _, row in forecast_df.iterrows()
        ],
    )


def _raise_model_error(exc: Exception):
    if isinstance(exc, ValueError):
        raise HTTPException(status_code=400, detail=str(exc))
    raise HTTPException(status_code=500, detail=f"Ошибка модели: {exc}")


@router.post("/forecast", response_model=ForecastResponse)
def forecast(request: ForecastRequest = Body(...)):
    try:
        return _forecast_response(request)
    except Exception as e:
        _raise_model_error(e)


@router.post("/forward", response_model=ForecastResponse)
def forward(request: ForecastRequest = Body(...)):
    try:
        return _forecast_response(request)
    except Exception as e:
        _raise_model_error(e)


@router.post("/evaluate", response_model=EvaluationResponse)
def evaluate(request: EvaluationRequest = Body(...)):
    try:
        series_df = get_series(request.ticker, request.target)
        series_df = series_df.sort_values("Date").reset_index(drop=True)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if len(series_df) <= request.test_size + 30:
        raise HTTPException(status_code=400, detail="not enough data for selected test_size")

    train_df = series_df.iloc[:-request.test_size].copy()
    test_df = series_df.iloc[-request.test_size:].copy()

    try:
        preds = forecast_series(
            train_df["value"].dropna(),
            request.test_size,
            request.model,
            request.rnn_type,
        )
    except Exception as e:
        _raise_model_error(e)

    actual = test_df["value"].astype(float).to_numpy()
    pred = np.asarray(preds, dtype=float)

    mae = float(np.mean(np.abs(actual - pred)))
    rmse = float(np.sqrt(np.mean((actual - pred) ** 2)))
    denominator = np.where(actual == 0, np.nan, actual)
    mape_value = np.nanmean(np.abs((actual - pred) / denominator)) * 100
    mape = None if pd.isna(mape_value) else float(mape_value)

    prediction = [
        Point(date=date.date().isoformat(), value=float(value))
        for date, value in zip(test_df["Date"], pred)
    ]

    return EvaluationResponse(
        ticker=request.ticker,
        target=request.target,
        model=request.model,
        rnn_type=request.rnn_type if request.model == "rnn" else None,
        train_size=len(train_df),
        test_size=len(test_df),
        metrics=Metrics(mae=mae, rmse=rmse, mape=mape),
        train=[
            Point(date=row["Date"].date().isoformat(), value=float(row["value"]))
            for _, row in train_df.tail(300).iterrows()
        ],
        test=[
            Point(date=row["Date"].date().isoformat(), value=float(row["value"]))
            for _, row in test_df.iterrows()
        ],
        prediction=prediction,
    )
