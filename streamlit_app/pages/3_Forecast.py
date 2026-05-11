from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils import MODEL_OPTIONS, download_dataframe, fetch_json, selection_payload


st.set_page_config(page_title="Future Forecast", layout="wide")
st.title("Future forecast")

ctx = selection_payload()
st.caption(f"{ctx['ticker']} · {ctx['target']}")

last_payload = st.session_state.get("last_model_payload", {})
default_model = last_payload.get("model", "naive")
default_index = MODEL_OPTIONS.index(default_model) if default_model in MODEL_OPTIONS else 0

left, middle, right = st.columns(3)

with left:
    model = st.selectbox("Model", MODEL_OPTIONS, index=default_index)

with middle:
    horizon = st.number_input("Horizon", min_value=1, max_value=365, value=30)

rnn_type = last_payload.get("rnn_type", "lstm")
with right:
    if model == "rnn":
        rnn_type = st.selectbox("RNN type", ["lstm", "gru"], index=0 if rnn_type == "lstm" else 1)
    else:
        st.empty()

if st.button("Построить прогноз", type="primary"):
    payload = {
        "ticker": ctx["ticker"],
        "target": ctx["target"],
        "horizon": int(horizon),
        "model": model,
    }
    if model == "rnn":
        payload["rnn_type"] = rnn_type

    with st.spinner("Строим прогноз..."):
        result = fetch_json("/forecast", method="POST", body=payload)

    st.session_state["last_forecast"] = result

result = st.session_state.get("last_forecast")

if result:
    hist = pd.DataFrame(result["history"])
    forecast = pd.DataFrame(result["forecast"])
    if not hist.empty:
        hist["date"] = pd.to_datetime(hist["date"])
    if not forecast.empty:
        forecast["date"] = pd.to_datetime(forecast["date"])

    fig = go.Figure()
    if not hist.empty:
        fig.add_trace(go.Scatter(x=hist["date"], y=hist["value"], mode="lines", name="history"))
    if not forecast.empty:
        fig.add_trace(go.Scatter(x=forecast["date"], y=forecast["value"], mode="lines", name="forecast"))

    fig.update_layout(
        title=f"{result['ticker']} - future forecast",
        hovermode="x unified",
        xaxis_title="Date",
        yaxis_title=result["target"],
    )
    fig.update_xaxes(rangeslider_visible=True)
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(forecast, use_container_width=True)
    download_dataframe(forecast, f"{result['ticker']}_{result['target']}_{result['model']}_forecast.csv")
