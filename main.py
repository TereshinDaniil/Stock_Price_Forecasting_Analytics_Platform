from __future__ import annotations

from fastapi import FastAPI

from api.routers import data, forecast, meta


app = FastAPI(
    title="Time Series API",
    description="Data and forecasting service",
)


@app.get("/")
def root():
    return {}


app.include_router(meta.router)
app.include_router(data.router)
app.include_router(forecast.router)
