from __future__ import annotations

from fastapi import FastAPI, Query, HTTPException, Body
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel, Field
from typing import Literal, Optional

import pandas as pd
import numpy as np

from services.plots.close_price import plot_close_price
from services.plots.returns import plot_daily_returns
from services.plots.structural_breaks import plot_structural_breaks
from services.plots.correlation import plot_feature_correlation
from services.plots.acf_pacf import plot_acf_pacf
from services.plots.lag_plot import plot_lag_plot

from services.models.forecast import run_naive_model
from services.stationarity import check_stationarity


app = FastAPI(
    title="Time Series API",
    description="EDA & Forecasting service"
)


class ForwardRequest(BaseModel):
    ticker: str
    target: str = "Close"
    horizon: int = Field(..., ge=1, le=365)
    model: Literal[
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
    ] = "naive"
    rnn_type: Literal["lstm", "gru"] = "lstm"


def add_percentage_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    cols = ["Open", "High", "Low", "Close", "Volume"]

    for col in cols:
        if col in df.columns:
            df[f"{col}_perc"] = (
                df.groupby("Ticker")[col]
                .pct_change() * 100
            )

    return df


def load_day_df() -> pd.DataFrame:
    df = pd.read_parquet("data/day_data.parquet")
    if "Date" not in df.columns or "Ticker" not in df.columns:
        raise HTTPException(status_code=500, detail="day_data.parquet must contain Date and Ticker")

    df["Date"] = pd.to_datetime(df["Date"], utc=True, errors="coerce")
    df = add_percentage_features(df)

    return df


def filter_ticker_range(
    df: pd.DataFrame,
    ticker: str,
    start_date: Optional[str],
    end_date: Optional[str],
) -> pd.DataFrame:
    g = df[df["Ticker"] == ticker].copy()
    if g.empty:
        return g

    if start_date is not None:
        start = pd.to_datetime(start_date, utc=True, errors="raise")
        g = g[g["Date"] >= start]

    if end_date is not None:
        end = pd.to_datetime(end_date, utc=True, errors="raise") + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
        g = g[g["Date"] <= end]

    return g.sort_values("Date")


def require_lib(lib_name: str, pip_name: str | None = None):
    pip_name = pip_name or lib_name
    raise HTTPException(
        status_code=500,
        detail=f"Missing dependency '{lib_name}'. Install: pip install {pip_name}"
    )


@app.post("/forward", tags=["Forward"], summary="Universal inference endpoint")
def forward(request: ForwardRequest = Body(...)):
    try:
        df_forecast = run_naive_model(
            ticker=request.ticker,
            target=request.target,
            horizon=request.horizon,
            model=request.model,
            rnn_type=request.rnn_type,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка модели: {e}")

    forecast_result = [
        {"date": row["Date"].strftime("%Y-%m-%d"), "value": float(row["Forecast"])}
        for _, row in df_forecast.iterrows()
    ]

    history_result: list[dict] = []
    history_window = int(request.horizon) * 5

    try:
        df = load_day_df()

        if request.target not in df.columns:
            raise HTTPException(status_code=400, detail=f"unknown target: {request.target}")

        g = (
            df[df["Ticker"] == request.ticker]
            .sort_values("Date")[["Date", request.target]]
            .dropna()
        )

        if not g.empty:
            hist = g.tail(history_window)
            history_result = [
                {"date": d.date().isoformat(), "value": float(v)}
                for d, v in zip(hist["Date"], hist[request.target])
            ]

    except HTTPException:
        raise
    except Exception:
        history_result = []

    return {
        "status": "ok",
        "ticker": request.ticker,
        "target": request.target,
        "model": request.model,
        "rnn_type": request.rnn_type if request.model == "rnn" else None,
        "horizon": request.horizon,
        "history_window": history_window,
        "history": history_result,
        "forecast": forecast_result
    }


@app.get("/meta/tickers", tags=["Meta"])
def meta_tickers():
    df = load_day_df()
    return sorted(df["Ticker"].dropna().unique().tolist())


@app.get("/meta/features", tags=["Meta"])
def meta_features():
    df = load_day_df()
    exclude = {"Date", "Ticker"}
    return sorted(
        c for c in df.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
    )


@app.get("/", tags=["UI"])
def home():
    return FileResponse("static/index.html")


@app.get("/eda/close-price", tags=["EDA"])
def close_price_png(ticker: str, start_date: str, end_date: str):
    try:
        buf = plot_close_price(ticker, start_date, end_date)
        return StreamingResponse(buf, media_type="image/png")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/eda/returns", tags=["EDA"])
def returns_png(ticker: str, start_date: str, end_date: str):
    try:
        buf = plot_daily_returns(ticker, start_date, end_date)
        return StreamingResponse(buf, media_type="image/png")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/eda/structural-breaks", tags=["EDA"])
def structural_breaks_png(
    ticker: str,
    start_date: str,
    end_date: str,
    kernel_n_bkps: int = Query(5, ge=1, le=20),
    pelt_penalty: int = Query(5, ge=1, le=50),
):
    try:
        buf = plot_structural_breaks(ticker, start_date, end_date, kernel_n_bkps, pelt_penalty)
        return StreamingResponse(buf, media_type="image/png")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/eda/correlation", tags=["EDA"])
def correlation_png(ticker: str):
    try:
        buf = plot_feature_correlation(ticker)
        return StreamingResponse(buf, media_type="image/png")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/eda/acf-pacf", tags=["EDA"])
def acf_pacf_png(
    ticker: str,
    feature: str,
    lags: int = Query(40, ge=10, le=200),
):
    try:
        df = load_day_df()
        stationarity_df = check_stationarity(df=df, target_cols=[feature], test="ADF")
        buf = plot_acf_pacf(ticker, feature, lags, stationarity_df)
        return StreamingResponse(buf, media_type="image/png")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/eda/lag-plot", tags=["EDA"])
def lag_plot_png(
    ticker: str,
    feature: str,
    lag: int = Query(1, ge=1, le=500),
):
    try:
        buf = plot_lag_plot(ticker, feature, lag)
        return StreamingResponse(buf, media_type="image/png")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/data/close-price", tags=["DATA"], summary="Close price as JSON for interactive plots")
def close_price_data(ticker: str, start_date: str, end_date: str):
    df = load_day_df()
    if "Close" not in df.columns:
        raise HTTPException(status_code=500, detail="day_data.parquet must contain Close")

    try:
        g = filter_ticker_range(df, ticker, start_date, end_date)
    except Exception:
        raise HTTPException(status_code=400, detail="start_date/end_date must be YYYY-MM-DD")

    if g.empty:
        raise HTTPException(status_code=400, detail="no data for given ticker/date range")

    g = g[["Date", "Close"]].dropna()
    points = [{"date": d.date().isoformat(), "value": float(v)} for d, v in zip(g["Date"], g["Close"])]

    return {"status": "ok", "ticker": ticker, "target": "Close", "start_date": start_date, "end_date": end_date, "points": points}


@app.get("/data/returns", tags=["DATA"], summary="Daily returns as JSON")
def returns_data(ticker: str, start_date: str, end_date: str):
    df = load_day_df()
    if "Close" not in df.columns:
        raise HTTPException(status_code=500, detail="day_data.parquet must contain Close")

    try:
        g = filter_ticker_range(df, ticker, start_date, end_date)
    except Exception:
        raise HTTPException(status_code=400, detail="start_date/end_date must be YYYY-MM-DD")

    if g.empty:
        raise HTTPException(status_code=400, detail="no data for given ticker/date range")

    g = g[["Date", "Close"]].dropna()
    g["return"] = g["Close"].pct_change()
    g = g.dropna(subset=["return"])

    points = [{"date": d.date().isoformat(), "value": float(v)} for d, v in zip(g["Date"], g["return"])]
    return {"status": "ok", "ticker": ticker, "target": "Close", "points": points}


@app.get("/data/correlation", tags=["DATA"], summary="Feature correlation matrix as JSON")
def correlation_data(ticker: str):
    df = load_day_df()
    g = df[df["Ticker"] == ticker].copy()
    if g.empty:
        raise HTTPException(status_code=400, detail="no data for ticker")

    exclude = {"Date", "Ticker"}
    num_cols = [c for c in g.columns if c not in exclude and pd.api.types.is_numeric_dtype(g[c])]
    if len(num_cols) < 2:
        raise HTTPException(status_code=400, detail="not enough numeric features for correlation")

    mat = g[num_cols].corr().fillna(0.0)
    return {"status": "ok", "ticker": ticker, "features": num_cols, "matrix": mat.values.tolist()}


@app.get("/data/acf-pacf", tags=["DATA"], summary="ACF/PACF values as JSON")
def acf_pacf_data(
    ticker: str,
    feature: str,
    lags: int = Query(40, ge=10, le=200),
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
):
    try:
        from statsmodels.tsa.stattools import acf, pacf
    except Exception:
        require_lib("statsmodels", "statsmodels")

    df = load_day_df()
    if feature not in df.columns:
        raise HTTPException(status_code=400, detail=f"unknown feature: {feature}")

    try:
        g = filter_ticker_range(df, ticker, start_date, end_date)
    except Exception:
        raise HTTPException(status_code=400, detail="start_date/end_date must be YYYY-MM-DD")

    if g.empty:
        raise HTTPException(status_code=400, detail="no data for ticker/date range")

    x = g[feature].dropna().astype(float).values
    if len(x) < max(30, int(lags) + 5):
        raise HTTPException(status_code=400, detail="not enough data for acf/pacf")

    a = acf(x, nlags=int(lags), fft=True)
    p = pacf(x, nlags=int(lags), method="ywm")

    return {
        "status": "ok",
        "ticker": ticker,
        "feature": feature,
        "lags": list(range(len(a))),
        "acf": [float(v) for v in a],
        "pacf": [float(v) for v in p]
    }


@app.get("/data/lag-plot", tags=["DATA"], summary="Lag plot points as JSON")
def lag_plot_data(
    ticker: str,
    feature: str,
    lag: int = Query(1, ge=1, le=500),
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
):
    df = load_day_df()
    if feature not in df.columns:
        raise HTTPException(status_code=400, detail=f"unknown feature: {feature}")

    try:
        g = filter_ticker_range(df, ticker, start_date, end_date)
    except Exception:
        raise HTTPException(status_code=400, detail="start_date/end_date must be YYYY-MM-DD")

    if g.empty:
        raise HTTPException(status_code=400, detail="no data for ticker/date range")

    s = g[feature].dropna().astype(float)
    lag = int(lag)
    if len(s) <= lag + 5:
        raise HTTPException(status_code=400, detail="not enough data for lag plot")

    x = s.iloc[:-lag].to_numpy()
    y = s.iloc[lag:].to_numpy()

    max_n = 20000
    if len(x) > max_n:
        idx = np.linspace(0, len(x) - 1, max_n).astype(int)
        x = x[idx]
        y = y[idx]

    points = [{"x": float(xi), "y": float(yi)} for xi, yi in zip(x, y)]
    return {"status": "ok", "ticker": ticker, "feature": feature, "lag": lag, "points": points}


@app.get("/data/structural-breaks", tags=["DATA"], summary="Structural breaks as JSON")
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
        require_lib("ruptures", "ruptures")

    df = load_day_df()
    if "Close" not in df.columns:
        raise HTTPException(status_code=500, detail="day_data.parquet must contain Close")

    try:
        g = filter_ticker_range(df, ticker, start_date, end_date)
    except Exception:
        raise HTTPException(status_code=400, detail="start_date/end_date must be YYYY-MM-DD")

    g = g[["Date", "Close"]].dropna()
    if len(g) < 50:
        raise HTTPException(status_code=400, detail="not enough data for structural breaks")

    y = g["Close"].astype(float).values.reshape(-1, 1)

    algo_k = rpt.KernelCPD(kernel="rbf").fit(y)
    bkps_k = algo_k.predict(n_bkps=int(kernel_n_bkps))

    algo_p = rpt.Pelt(model="rbf").fit(y)
    bkps_p = algo_p.predict(pen=int(pelt_penalty))

    def idx_to_dates(bkps: list[int]) -> list[str]:
        idxs = [i for i in bkps if i < len(g)]
        return [g.iloc[i]["Date"].date().isoformat() for i in idxs]

    points = [{"date": d.date().isoformat(), "value": float(v)} for d, v in zip(g["Date"], g["Close"])]

    return {
        "status": "ok",
        "ticker": ticker,
        "target": "Close",
        "start_date": start_date,
        "end_date": end_date,
        "points": points,
        "kernel_breaks": idx_to_dates(bkps_k),
        "pelt_breaks": idx_to_dates(bkps_p),
        "kernel_n_bkps": int(kernel_n_bkps),
        "pelt_penalty": int(pelt_penalty),
    }
