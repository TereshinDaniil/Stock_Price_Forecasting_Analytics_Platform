from __future__ import annotations

import os
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

FASTAPI_BASE = os.getenv("FASTAPI_BASE", "http://127.0.0.1:8000")
PERCENT_FEATURES = ["Open_perc", "High_perc", "Low_perc", "Close_perc", "Volume_perc"]
MODEL_OPTIONS = [
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


def fetch_json(path: str, params: Optional[dict] = None, method: str = "GET", body: Optional[dict] = None):
    url = f"{FASTAPI_BASE}{path}"
    try:
        if method == "GET":
            response = requests.get(url, params=params, timeout=180)
        else:
            response = requests.post(url, json=body, timeout=180)
    except Exception as e:
        st.error(f"Не удалось подключиться к FastAPI: {e}")
        st.stop()
        raise RuntimeError("FastAPI connection failed") from e

    if response.status_code != 200:
        st.error(f"Ошибка FastAPI: {response.status_code}")
        content_type = (response.headers.get("content-type") or "").lower()
        if "application/json" in content_type:
            try:
                st.json(response.json())
            except Exception:
                st.text(response.text)
        else:
            st.text(response.text)
        st.stop()
        raise RuntimeError(f"FastAPI returned status {response.status_code}")

    return response.json()


@st.cache_data
def get_tickers():
    return fetch_json("/meta/tickers")


@st.cache_data
def get_features():
    features = fetch_json("/meta/features")
    for feature in PERCENT_FEATURES:
        if feature not in features:
            features.append(feature)
    return features


@st.cache_data
def get_ticker_date_range(ticker: str):
    df = pd.read_parquet(PROJECT_ROOT / "data/day_data.parquet")
    df["Date"] = pd.to_datetime(df["Date"], utc=True, errors="coerce")
    result = df[df["Ticker"] == ticker].dropna(subset=["Date"])

    if result.empty:
        end = date.today()
        return end - timedelta(days=365), end

    return result["Date"].min().date(), result["Date"].max().date()


def require_selection():
    missing = [
        key
        for key in ["ticker", "target", "start_date", "end_date"]
        if key not in st.session_state
    ]
    if missing:
        st.warning("Сначала выберите данные на стартовой странице.")
        st.page_link("app.py", label="Перейти к выбору данных")
        st.stop()


def selection_payload() -> dict:
    require_selection()
    return {
        "ticker": st.session_state["ticker"],
        "target": st.session_state["target"],
        "start_date": st.session_state["start_date"],
        "end_date": st.session_state["end_date"],
    }


def dataframe_from_points(points: list[dict], value_name: str = "value") -> pd.DataFrame:
    df = pd.DataFrame(points)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    return df.rename(columns={"value": value_name}).sort_values("date")


def download_dataframe(df: pd.DataFrame, file_name: str, label: str = "Скачать CSV"):
    st.download_button(
        label,
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=file_name,
        mime="text/csv",
    )
