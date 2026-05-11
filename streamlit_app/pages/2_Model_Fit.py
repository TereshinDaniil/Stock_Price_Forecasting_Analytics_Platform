from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils import MODEL_OPTIONS, download_dataframe, fetch_json, selection_payload


st.set_page_config(page_title="Fit Model", layout="wide")
st.title("Fit model")

ctx = selection_payload()
st.caption(f"{ctx['ticker']} · {ctx['target']} · train/test evaluation")

left, right, extra = st.columns(3)

with left:
    model = st.selectbox("Model", MODEL_OPTIONS, index=0)

with right:
    test_size = st.number_input("Test size", min_value=1, max_value=365, value=30)

rnn_type = "lstm"
with extra:
    if model == "rnn":
        rnn_type = st.selectbox("RNN type", ["lstm", "gru"], index=0)
    else:
        st.empty()

if model in {"mlp", "rnn", "chronos"}:
    st.info("Нейросетевые и предобученные модели могут считаться дольше классических моделей.")

if st.button("Fit and evaluate", type="primary"):
    payload = {
        "ticker": ctx["ticker"],
        "target": ctx["target"],
        "model": model,
        "test_size": int(test_size),
    }
    if model == "rnn":
        payload["rnn_type"] = rnn_type

    with st.spinner("Обучаем модель и считаем метрики..."):
        result = fetch_json("/evaluate", method="POST", body=payload)

    st.session_state["last_model_payload"] = payload
    st.session_state["last_evaluation"] = result

result = st.session_state.get("last_evaluation")

if result:
    metrics = result["metrics"]
    c1, c2, c3 = st.columns(3)
    c1.metric("MAE", f"{metrics['mae']:.4f}")
    c2.metric("RMSE", f"{metrics['rmse']:.4f}")
    c3.metric("MAPE", "n/a" if metrics["mape"] is None else f"{metrics['mape']:.2f}%")

    train = pd.DataFrame(result["train"])
    test = pd.DataFrame(result["test"])
    pred = pd.DataFrame(result["prediction"])
    for df in [train, test, pred]:
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"])

    fig = go.Figure()
    if not train.empty:
        fig.add_trace(go.Scatter(x=train["date"], y=train["value"], mode="lines", name="train"))
    if not test.empty:
        fig.add_trace(go.Scatter(x=test["date"], y=test["value"], mode="lines", name="fact"))
    if not pred.empty:
        fig.add_trace(go.Scatter(x=pred["date"], y=pred["value"], mode="lines", name="prediction"))

    fig.update_layout(
        title=f"{result['ticker']} - fact vs prediction",
        hovermode="x unified",
        xaxis_title="Date",
        yaxis_title=result["target"],
    )
    fig.update_xaxes(rangeslider_visible=True)
    st.plotly_chart(fig, use_container_width=True)

    table = test.rename(columns={"value": "actual"}).merge(
        pred.rename(columns={"value": "prediction"}),
        on="date",
        how="left",
    )
    st.dataframe(table, use_container_width=True)
    download_dataframe(table, f"{result['ticker']}_{result['target']}_{result['model']}_evaluation.csv")

    st.page_link("pages/3_Forecast.py", label="Перейти к прогнозу в будущее")
