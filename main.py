from fastapi import FastAPI, Query, HTTPException, Body
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel, Field
from typing import Literal
import pandas as pd

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
        "exp_smoothing"
    ] = "naive"


@app.post("/forward", tags=["Forward"], summary="Universal inference endpoint")
def forward(request: ForwardRequest = Body(...)):
    try:
        df_forecast = run_naive_model(
            ticker=request.ticker,
            target=request.target,
            horizon=request.horizon,
            model=request.model,
        )

    except Exception:
        raise HTTPException(
            status_code=403,
            detail="модель не смогла обработать данные"
        )

    result = [
        {
            "date": row["Date"].strftime("%Y-%m-%d"),
            "value": float(row["Forecast"])
        }
        for _, row in df_forecast.iterrows()
    ]

    return {
        "status": "ok",
        "ticker": request.ticker,
        "target": request.target,
        "model": request.model,
        "horizon": request.horizon,
        "forecast": result
    }

@app.get("/meta/tickers")
def meta_tickers():
    df = pd.read_parquet("data/day_data.parquet")
    return sorted(df["Ticker"].dropna().unique().tolist())


@app.get("/meta/features")
def meta_features():
    df = pd.read_parquet("data/day_data.parquet")
    exclude = {"Date", "Ticker"}
    return sorted(
        c for c in df.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
    )

@app.get("/")
def home():
    return FileResponse("static/index.html")


@app.get("/eda/close-price", tags=["EDA"])
def close_price(
    ticker: str,
    start_date: str,
    end_date: str,
):
    try:
        buf = plot_close_price(ticker, start_date, end_date)
        return StreamingResponse(buf, media_type="image/png")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/eda/returns", tags=["EDA"])
def daily_returns(
    ticker: str,
    start_date: str,
    end_date: str,
):
    try:
        buf = plot_daily_returns(ticker, start_date, end_date)
        return StreamingResponse(buf, media_type="image/png")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/eda/structural-breaks", tags=["EDA"])
def structural_breaks(
    ticker: str,
    start_date: str,
    end_date: str,
    kernel_n_bkps: int = Query(5, ge=1, le=20),
    pelt_penalty: int = Query(5, ge=1, le=50),
):
    try:
        buf = plot_structural_breaks(
            ticker,
            start_date,
            end_date,
            kernel_n_bkps,
            pelt_penalty
        )
        return StreamingResponse(buf, media_type="image/png")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/eda/correlation", tags=["EDA"])
def correlation(ticker: str):
    try:
        buf = plot_feature_correlation(ticker)
        return StreamingResponse(buf, media_type="image/png")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/eda/acf-pacf", tags=["EDA"])
def acf_pacf(
    ticker: str,
    feature: str,
    lags: int = Query(40, ge=10, le=200),
):
    try:
        df = pd.read_parquet("data/day_data.parquet")

        stationarity_df = check_stationarity(
            df=df,
            target_cols=[feature],
            test="ADF"
        )

        buf = plot_acf_pacf(
            ticker,
            feature,
            lags,
            stationarity_df
        )
        return StreamingResponse(buf, media_type="image/png")

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/eda/lag-plot", tags=["EDA"])
def lag_plot(
    ticker: str,
    feature: str,
    lag: int = Query(1, ge=1, le=500),
):
    try:
        buf = plot_lag_plot(ticker, feature, lag)
        return StreamingResponse(buf, media_type="image/png")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


