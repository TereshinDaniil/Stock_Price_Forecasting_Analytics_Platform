from __future__ import annotations

from fastapi import APIRouter, HTTPException

from services.data_service import load_day_df, numeric_features, tickers


router = APIRouter(prefix="/meta", tags=["Meta"])


@router.get("/tickers", response_model=list[str])
def meta_tickers():
    try:
        return tickers()
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/features", response_model=list[str])
def meta_features():
    try:
        return numeric_features(load_day_df())
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))
