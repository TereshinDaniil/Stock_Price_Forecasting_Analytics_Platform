from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils import dataframe_from_points, download_dataframe, fetch_json, selection_payload


st.set_page_config(page_title="EDA", layout="wide")
st.title("EDA")

ctx = selection_payload()
st.caption(f"{ctx['ticker']} · {ctx['target']} · {ctx['start_date']} - {ctx['end_date']}")

graph_type = st.selectbox(
    "Инструмент",
    [
        "Series",
        "Returns",
        "Correlation",
        "ACF/PACF",
        "Lag plot",
        "Structural breaks",
    ],
)

if graph_type == "Series":
    data = fetch_json(
        "/data/series",
        params={**ctx, "kind": "value"},
    )
    df = dataframe_from_points(data["points"], ctx["target"])
    fig = go.Figure(go.Scatter(x=df["date"], y=df[ctx["target"]], mode="lines", name=ctx["target"]))
    fig.update_layout(title=f"{ctx['ticker']} - {ctx['target']}", hovermode="x unified")
    fig.update_xaxes(rangeslider_visible=True)
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(df, use_container_width=True)
    download_dataframe(df, f"{ctx['ticker']}_{ctx['target']}_series.csv")

elif graph_type == "Returns":
    data = fetch_json(
        "/data/series",
        params={**ctx, "kind": "returns"},
    )
    df = dataframe_from_points(data["points"], "return")
    fig = go.Figure(go.Scatter(x=df["date"], y=df["return"] * 100, mode="lines", name="Return (%)"))
    fig.update_layout(title=f"{ctx['ticker']} - returns", yaxis_title="Return (%)", hovermode="x unified")
    fig.update_xaxes(rangeslider_visible=True)
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(df, use_container_width=True)
    download_dataframe(df, f"{ctx['ticker']}_{ctx['target']}_returns.csv")

elif graph_type == "Correlation":
    data = fetch_json("/data/correlation", params={"ticker": ctx["ticker"]})
    matrix = pd.DataFrame(data["matrix"], index=data["features"], columns=data["features"])
    fig = px.imshow(matrix, aspect="auto", title=f"{ctx['ticker']} - correlation")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(matrix, use_container_width=True)
    download_dataframe(matrix.reset_index(names="feature"), f"{ctx['ticker']}_correlation.csv")

elif graph_type == "ACF/PACF":
    lags = st.number_input("Lags", min_value=10, max_value=200, value=40)
    data = fetch_json(
        "/data/acf-pacf",
        params={
            "ticker": ctx["ticker"],
            "feature": ctx["target"],
            "lags": int(lags),
            "start_date": ctx["start_date"],
            "end_date": ctx["end_date"],
        },
    )
    df = pd.DataFrame({"lag": data["lags"], "acf": data["acf"], "pacf": data["pacf"]})
    left, right = st.columns(2)
    with left:
        st.plotly_chart(go.Figure(go.Bar(x=df["lag"], y=df["acf"], name="ACF")), use_container_width=True)
    with right:
        st.plotly_chart(go.Figure(go.Bar(x=df["lag"], y=df["pacf"], name="PACF")), use_container_width=True)
    st.dataframe(df, use_container_width=True)
    download_dataframe(df, f"{ctx['ticker']}_{ctx['target']}_acf_pacf.csv")

elif graph_type == "Lag plot":
    lag = st.number_input("Lag", min_value=1, max_value=500, value=1)
    data = fetch_json(
        "/data/lag-plot",
        params={
            "ticker": ctx["ticker"],
            "feature": ctx["target"],
            "lag": int(lag),
            "start_date": ctx["start_date"],
            "end_date": ctx["end_date"],
        },
    )
    df = pd.DataFrame(data["points"])
    fig = go.Figure(go.Scatter(x=df["x"], y=df["y"], mode="markers", name=f"lag={lag}"))
    fig.update_layout(xaxis_title=f"{ctx['target']}(t)", yaxis_title=f"{ctx['target']}(t+lag)")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(df, use_container_width=True)
    download_dataframe(df, f"{ctx['ticker']}_{ctx['target']}_lag_{lag}.csv")

elif graph_type == "Structural breaks":
    left, right = st.columns(2)
    with left:
        kernel_n_bkps = st.number_input("Kernel breakpoints", min_value=1, max_value=20, value=5)
    with right:
        pelt_penalty = st.number_input("PELT penalty", min_value=1, max_value=50, value=5)

    data = fetch_json(
        "/data/structural-breaks",
        params={
            "ticker": ctx["ticker"],
            "start_date": ctx["start_date"],
            "end_date": ctx["end_date"],
            "kernel_n_bkps": int(kernel_n_bkps),
            "pelt_penalty": int(pelt_penalty),
        },
    )
    df = dataframe_from_points(data["points"], "Close")
    fig = go.Figure(go.Scatter(x=df["date"], y=df["Close"], mode="lines", name="Close"))
    for break_date in data["kernel_breaks"]:
        fig.add_vline(x=pd.to_datetime(break_date), line_width=1)
    for break_date in data["pelt_breaks"]:
        fig.add_vline(x=pd.to_datetime(break_date), line_width=1, line_dash="dash")
    fig.update_layout(title="Structural breaks", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    breaks = pd.DataFrame(
        {
            "kernel_breaks": pd.Series(data["kernel_breaks"]),
            "pelt_breaks": pd.Series(data["pelt_breaks"]),
        }
    )
    st.dataframe(breaks, use_container_width=True)
    download_dataframe(df, f"{ctx['ticker']}_structural_breaks_series.csv")
    download_dataframe(breaks, f"{ctx['ticker']}_structural_breaks.csv", label="Скачать даты разрывов")
