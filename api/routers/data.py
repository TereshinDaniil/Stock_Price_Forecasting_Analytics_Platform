from __future__ import annotations

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from api.schemas import (
    AcfPacfResponse,
    CorrelationResponse,
    LagPlotResponse,
    Point,
    SeriesResponse,
    StructuralBreaksResponse,
    XYPoint,
)
from services.data_service import filter_ticker_range, get_series, load_day_df


router = APIRouter(prefix="/data", tags=["Data"])


def _points_from_df(df: pd.DataFrame) -> list[Point]:
    return [
        Point(date=row["Date"].date().isoformat(), value=float(row["value"]))
        for _, row in df.iterrows()
    ]


@router.get("/series", response_model=SeriesResponse)
def series_data(
    ticker: str,
    target: str = "Close",
    kind: str = Query("value", pattern="^(value|returns)$"),
    start_date: str | None = None,
    end_date: str | None = None,
):
    try:
        series = get_series(ticker, target, start_date, end_date)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        raise HTTPException(status_code=400, detail="start_date/end_date must be YYYY-MM-DD")

    if kind == "returns":
        series["value"] = series["value"].pct_change()
        series = series.dropna(subset=["value"])

    return SeriesResponse(
        ticker=ticker,
        target=target,
        kind=kind,
        points=_points_from_df(series),
    )


@router.get("/correlation", response_model=CorrelationResponse)
def correlation_data(ticker: str):
    df = load_day_df()
    result = df[df["Ticker"] == ticker].copy()
    if result.empty:
        raise HTTPException(status_code=400, detail="no data for ticker")

    exclude = {"Date", "Ticker"}
    features = [
        col
        for col in result.columns
        if col not in exclude and pd.api.types.is_numeric_dtype(result[col])
    ]
    if len(features) < 2:
        raise HTTPException(status_code=400, detail="not enough numeric features for correlation")

    matrix = result[features].corr().fillna(0.0)
    return CorrelationResponse(
        ticker=ticker,
        features=features,
        matrix=matrix.values.tolist(),
    )


@router.get("/acf-pacf", response_model=AcfPacfResponse)
def acf_pacf_data(
    ticker: str,
    feature: str,
    lags: int = Query(40, ge=10, le=200),
    start_date: str | None = None,
    end_date: str | None = None,
):
    try:
        from statsmodels.tsa.stattools import acf, pacf
    except Exception:
        raise HTTPException(status_code=500, detail="Missing dependency 'statsmodels'")

    try:
        series = get_series(ticker, feature, start_date, end_date)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    values = series["value"].astype(float).values
    if len(values) < max(30, int(lags) + 5):
        raise HTTPException(status_code=400, detail="not enough data for acf/pacf")

    acf_values = acf(values, nlags=int(lags), fft=True)
    pacf_values = pacf(values, nlags=int(lags), method="ywm")

    return AcfPacfResponse(
        ticker=ticker,
        feature=feature,
        lags=list(range(len(acf_values))),
        acf=[float(v) for v in acf_values],
        pacf=[float(v) for v in pacf_values],
    )


@router.get("/lag-plot", response_model=LagPlotResponse)
def lag_plot_data(
    ticker: str,
    feature: str,
    lag: int = Query(1, ge=1, le=500),
    start_date: str | None = None,
    end_date: str | None = None,
):
    try:
        series = get_series(ticker, feature, start_date, end_date)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    values = series["value"].dropna().astype(float)
    if len(values) <= lag + 5:
        raise HTTPException(status_code=400, detail="not enough data for lag plot")

    x = values.iloc[:-lag].to_numpy()
    y = values.iloc[lag:].to_numpy()

    max_n = 20000
    if len(x) > max_n:
        idx = np.linspace(0, len(x) - 1, max_n).astype(int)
        x = x[idx]
        y = y[idx]

    return LagPlotResponse(
        ticker=ticker,
        feature=feature,
        lag=lag,
        points=[XYPoint(x=float(xi), y=float(yi)) for xi, yi in zip(x, y)],
    )


@router.get("/structural-breaks", response_model=StructuralBreaksResponse)
def structural_breaks_data(
    ticker: str,
    start_date: str,
    end_date: str,
    kernel_n_bkps: int = Query(5, ge=1, le=20),
    pelt_penalty: int = Query(5, ge=1, le=50),
):
    try:
        import ruptures as rpt
    except Exception:
        raise HTTPException(status_code=500, detail="Missing dependency 'ruptures'")

    df = load_day_df()
    try:
        result = filter_ticker_range(df, ticker, start_date, end_date)
    except Exception:
        raise HTTPException(status_code=400, detail="start_date/end_date must be YYYY-MM-DD")

    if "Close" not in result.columns:
        raise HTTPException(status_code=500, detail="day_data.parquet must contain Close")

    result = result[["Date", "Close"]].dropna().rename(columns={"Close": "value"})
    if len(result) < 50:
        raise HTTPException(status_code=400, detail="not enough data for structural breaks")

    values = result["value"].astype(float).values.reshape(-1, 1)

    kernel = rpt.KernelCPD(kernel="rbf").fit(values)
    kernel_breaks = kernel.predict(n_bkps=int(kernel_n_bkps))

    pelt = rpt.Pelt(model="rbf").fit(values)
    pelt_breaks = pelt.predict(pen=int(pelt_penalty))

    def idx_to_dates(breaks: list[int]) -> list[str]:
        idxs = [i for i in breaks if i < len(result)]
        return [result.iloc[i]["Date"].date().isoformat() for i in idxs]

    return StructuralBreaksResponse(
        ticker=ticker,
        points=_points_from_df(result),
        kernel_breaks=idx_to_dates(kernel_breaks),
        pelt_breaks=idx_to_dates(pelt_breaks),
        kernel_n_bkps=kernel_n_bkps,
        pelt_penalty=pelt_penalty,
    )
