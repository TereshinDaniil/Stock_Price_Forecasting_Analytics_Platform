import os
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from typing import Optional, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.outliers import detect_ohlcv_outliers

FASTAPI_BASE = os.getenv("FASTAPI_BASE", "http://127.0.0.1:8000")

st.set_page_config(page_title="Time Series UI", layout="wide")
st.title("Time Series: EDA + Forecast")

PERCENT_FEATURES = ["Open_perc", "High_perc", "Low_perc", "Close_perc", "Volume_perc"]
OHLC_FEATURES = ["Open", "High", "Low", "Close"]


# helpers
def fetch_json(path: str, params: Optional[dict] = None, method: str = "GET", body: Optional[dict] = None):
    url = f"{FASTAPI_BASE}{path}"
    timeout = 180
    try:
        if method == "GET":
            r = requests.get(url, params=params, timeout=timeout)
        else:
            r = requests.post(url, json=body, timeout=timeout)
    except Exception as e:
        st.error(f"Не удалось подключиться к FastAPI: {e}")
        st.stop()
        raise RuntimeError("FastAPI connection failed") from e

    if r.status_code != 200:
        st.error(f"Ошибка FastAPI: {r.status_code}")
        ct = (r.headers.get("content-type") or "").lower()
        if "application/json" in ct:
            try:
                st.json(r.json())
            except Exception:
                st.text(r.text)
        else:
            st.text(r.text)
        st.stop()
        raise RuntimeError(f"FastAPI returned status {r.status_code}")

    return r.json()


def cache_key(prefix: str, params: dict) -> str:
    items = sorted((k, str(v)) for k, v in params.items())
    return prefix + "|" + "|".join([f"{k}={v}" for k, v in items])


def _pick_outlier_flag_col(df: pd.DataFrame) -> Optional[str]:
    if "any_outlier" in df.columns:
        return "any_outlier"
    if "price_outlier" in df.columns:
        return "price_outlier"
    return None


def _segment_traces(df: pd.DataFrame, break_dates: list[str], x_col: str, y_col: str, label_prefix: str):
    if df.empty:
        return []

    dfx = df.copy()
    dfx[x_col] = pd.to_datetime(dfx[x_col])
    dfx = dfx.sort_values(x_col).reset_index(drop=True)

    bd = pd.to_datetime(break_dates, errors="coerce")
    bd = bd[~bd.isna()]
    if len(bd) == 0:
        return [go.Scatter(x=dfx[x_col], y=dfx[y_col], mode="lines", name=label_prefix)]

    min_x = dfx[x_col].min()
    max_x = dfx[x_col].max()
    bd = sorted([d for d in bd if min_x <= d <= max_x])

    break_idx = []
    for d in bd:
        idx = int(dfx[dfx[x_col] >= d].index.min()) if (dfx[x_col] >= d).any() else None
        if idx is not None and 0 < idx < len(dfx):
            break_idx.append(idx)

    break_idx = sorted(set(break_idx))
    cuts = [0] + break_idx + [len(dfx)]

    traces = []
    for i in range(len(cuts) - 1):
        a, b = cuts[i], cuts[i + 1]
        seg = dfx.iloc[a:b]
        if seg.empty:
            continue
        traces.append(
            go.Scatter(
                x=seg[x_col],
                y=seg[y_col],
                mode="lines",
                name=f"{label_prefix} seg {i + 1}",
                showlegend=False,
            )
        )

    traces.append(
        go.Scatter(
            x=[dfx[x_col].iloc[0]],
            y=[dfx[y_col].iloc[0]],
            mode="lines",
            name=label_prefix,
            showlegend=True,
            opacity=0
        )
    )

    return traces


@st.cache_data(ttl=300)
def get_tickers():
    return fetch_json("/meta/tickers")


@st.cache_data(ttl=300)
def get_features():
    return fetch_json("/meta/features")


@st.cache_data(ttl=300)
def get_ticker_date_range(ticker: str):
    df = pd.read_parquet("data/day_data.parquet")
    df["Date"] = pd.to_datetime(df["Date"], utc=True, errors="coerce")
    g = df[(df["Ticker"] == ticker)].dropna(subset=["Date"])

    if g.empty:
        end = date.today()
        start = end - timedelta(days=365)
        return start, end

    return g["Date"].min().date(), g["Date"].max().date()


@st.cache_data(ttl=300)
def load_day_feature_df(ticker: str, start_date_str: str, end_date_str: str, feature: str) -> pd.DataFrame:
    df = pd.read_parquet("data/day_data.parquet")
    df["Date"] = pd.to_datetime(df["Date"], utc=True, errors="coerce")

    start_dt = pd.to_datetime(start_date_str, utc=True)
    end_dt = pd.to_datetime(end_date_str, utc=True)

    df = df[
        (df["Ticker"] == ticker) &
        (df["Date"] >= start_dt) &
        (df["Date"] <= end_dt)
    ].copy()

    if df.empty or feature not in df.columns:
        return pd.DataFrame()

    # для классических ценовых колонок считаем выбросы
    if any(col in df.columns for col in ["Open", "High", "Low", "Close", "Volume"]):
        df = detect_ohlcv_outliers(df)

    df = df.sort_values("Date")
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
    df["value"] = pd.to_numeric(df[feature], errors="coerce")
    df = df.dropna(subset=["value"])

    return df[["Date", "Ticker", "value"] + [c for c in df.columns if c not in ["Date", "Ticker", "value"]]]


@st.cache_data(ttl=300)
def load_day_df_for_returns(ticker: str, start_date_str: str, end_date_str: str) -> pd.DataFrame:
    df = pd.read_parquet("data/day_data.parquet")
    df["Date"] = pd.to_datetime(df["Date"], utc=True, errors="coerce")

    start_dt = pd.to_datetime(start_date_str, utc=True)
    end_dt = pd.to_datetime(end_date_str, utc=True)

    df = df[
        (df["Ticker"] == ticker) &
        (df["Date"] >= start_dt) &
        (df["Date"] <= end_dt)
    ].copy()

    if df.empty:
        return df

    df = detect_ohlcv_outliers(df)

    if "Return" not in df.columns:
        if "Close" in df.columns:
            df = df.sort_values("Date")
            df["Return"] = df["Close"].pct_change()
        else:
            return pd.DataFrame()

    df["Return_pct"] = df["Return"] * 100
    df = df.sort_values("Date")
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
    return df


@st.cache_data(ttl=300)
def load_day_df_for_candles(ticker: str, start_date_str: str, end_date_str: str) -> pd.DataFrame:
    df = pd.read_parquet("data/day_data.parquet")
    df["Date"] = pd.to_datetime(df["Date"], utc=True, errors="coerce")

    start_dt = pd.to_datetime(start_date_str, utc=True)
    end_dt = pd.to_datetime(end_date_str, utc=True)

    df = df[
        (df["Ticker"] == ticker) &
        (df["Date"] >= start_dt) &
        (df["Date"] <= end_dt)
    ].copy()

    needed = {"Date", "Open", "High", "Low", "Close"}
    if df.empty or not needed.issubset(df.columns):
        return pd.DataFrame()

    df = df.sort_values("Date")
    df = detect_ohlcv_outliers(df)
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
    return df


# meta
tickers = get_tickers()
features = get_features()

# на случай, если бэк еще не отдает новые колонки
for feat in PERCENT_FEATURES:
    if feat not in features:
        features.append(feat)


# sidebar
with st.sidebar:
    st.header("Параметры")

    ticker = st.selectbox("Ticker", tickers, index=0 if tickers else None)

    ticker_min_date, ticker_max_date = get_ticker_date_range(ticker)

    default_end = ticker_max_date
    default_start = max(ticker_min_date, default_end - timedelta(days=365))

    start_date = st.date_input(
        "Start date",
        value=default_start,
        min_value=ticker_min_date,
        max_value=ticker_max_date
    )

    end_date = st.date_input(
        "End date",
        value=default_end,
        min_value=ticker_min_date,
        max_value=ticker_max_date
    )

    if start_date > end_date:
        st.error("Start date > End date")
        st.stop()

    start_date_str = start_date.strftime("%Y-%m-%d")
    end_date_str = end_date.strftime("%Y-%m-%d")


tab_forecast, tab_eda = st.tabs(["Forecast", "EDA"])


# Forecast
with tab_forecast:
    st.subheader("Forecast (/forward)")

    col1, col2, col3 = st.columns(3)

    with col1:
        default_target_idx = features.index("Close") if "Close" in features else 0
        target = st.selectbox("Target", features, index=default_target_idx)

    with col2:
        model = st.selectbox(
            "Model",
            ["naive", "seasonal_naive", "moving_average", "drift", "exp_smoothing", "random_forest", "linear", "mlp", "rnn", "chronos"],
            index=0
        )

    with col3:
        horizon = st.number_input("Horizon", min_value=1, max_value=365, value=30)

    rnn_type = "lstm"
    if model == "rnn":
        rnn_type = st.selectbox("RNN type", ["lstm", "gru"], index=0)

    if target in PERCENT_FEATURES:
        st.info("Выбран процентный признак. Прогноз будет строиться для процентных изменений, а не для абсолютной цены.")

    if st.button("Спрогнозировать", type="primary", key="run_forecast"):
        payload = {"ticker": ticker, "target": target, "horizon": int(horizon), "model": model}
        if model == "rnn":
            payload["rnn_type"] = rnn_type

        with st.expander("Payload"):
            st.json(payload)

        data = fetch_json("/forward", method="POST", body=payload)

        hist = pd.DataFrame(data.get("history", []))
        if not hist.empty:
            hist["date"] = pd.to_datetime(hist["date"])
            hist = hist.sort_values("date")

        fc = pd.DataFrame(data.get("forecast", []))
        if fc.empty:
            st.warning("Пустой forecast в ответе")
            st.json(data)
            st.stop()

        fc["date"] = pd.to_datetime(fc["date"])
        fc = fc.sort_values("date")

        y_title = f"{target}"

        fig = go.Figure()

        if not hist.empty:
            fig.add_trace(go.Scatter(
                x=hist["date"],
                y=hist["value"],
                mode="lines",
                name=f"history ({data.get('history_window', 'n/a')} pts)"
            ))

        fig.add_trace(go.Scatter(
            x=fc["date"],
            y=fc["value"],
            mode="lines",
            name="forecast"
        ))

        fig.update_layout(
            hovermode="x unified",
            xaxis_title="Date",
            yaxis_title=y_title,
            title=f"{ticker} — forecast for {target}"
        )
        fig.update_xaxes(rangeslider_visible=True)

        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Forecast table")
        st.dataframe(fc)

        st.download_button(
            "Скачать CSV",
            data=fc.to_csv(index=False).encode("utf-8"),
            file_name=f"{ticker}_{target}_{model}_h{horizon}.csv",
            mime="text/csv",
            key=f"dl_forecast_{ticker}_{target}_{model}_{horizon}"
        )


# EDA
with tab_eda:
    st.subheader("EDA")

    graph_type = st.selectbox(
        "Тип графика",
        [
            "Feature series",
            "Candles",
            "Returns (with outliers)",
            "Structural breaks (2 charts)",
            "Correlation heatmap",
            "ACF/PACF",
            "Lag plot",
        ],
        index=0,
        key="eda_graph_type"
    )

    controls_area = st.container()
    plot_area = st.container()

    with controls_area:
        needs_dates = graph_type in ["Feature series", "Candles", "Returns (with outliers)", "Structural breaks (2 charts)"]
        if needs_dates:
            st.caption(f"Дата-диапазон: {start_date_str} — {end_date_str}")

        feature_series_params = None
        sb_params = None
        acf_params = None
        lag_params = None

        if graph_type == "Feature series":
            feature_for_series = st.selectbox(
                "feature",
                features,
                index=features.index("Close") if "Close" in features else 0,
                key="series_feature"
            )
            feature_series_params = {
                "ticker": ticker,
                "start_date": start_date_str,
                "end_date": end_date_str,
                "feature": feature_for_series
            }

        if graph_type == "Structural breaks (2 charts)":
            c1, c2 = st.columns(2)
            with c1:
                kernel_n_bkps = st.number_input("kernel_n_bkps", 1, 20, 5, key="sb_kernel_n")
            with c2:
                pelt_penalty = st.number_input("pelt_penalty", 1, 50, 5, key="sb_pelt_pen")

            sb_params = {
                "ticker": ticker,
                "start_date": start_date_str,
                "end_date": end_date_str,
                "kernel_n_bkps": int(kernel_n_bkps),
                "pelt_penalty": int(pelt_penalty),
            }

        if graph_type == "ACF/PACF":
            feat = st.selectbox("feature", features, index=0 if features else None, key="acf_feature")
            lags = st.number_input("lags", 10, 200, 40, key="acf_lags")
            use_dates = st.checkbox("Ограничить по датам", value=False, key="acf_use_dates")

            acf_params = {"ticker": ticker, "feature": feat, "lags": int(lags)}
            if use_dates:
                acf_params["start_date"] = start_date_str
                acf_params["end_date"] = end_date_str

        if graph_type == "Lag plot":
            feat = st.selectbox("feature", features, index=0 if features else None, key="lag_feature")
            lag = st.number_input("lag", 1, 500, 1, key="lag_lag")
            use_dates = st.checkbox("Ограничить по датам", value=False, key="lag_use_dates")

            lag_params = {"ticker": ticker, "feature": feat, "lag": int(lag)}
            if use_dates:
                lag_params["start_date"] = start_date_str
                lag_params["end_date"] = end_date_str

        run_eda = st.button("Построить", type="primary", key="run_eda")

    if run_eda:
        if graph_type == "Feature series":
            params = feature_series_params
            key = cache_key("feature_series", params)
            st.session_state["eda_last"] = ("Feature series", key)
            if key not in st.session_state:
                st.session_state[key] = {
                    "feature": params["feature"],
                    "df": load_day_feature_df(
                        ticker=params["ticker"],
                        start_date_str=params["start_date"],
                        end_date_str=params["end_date"],
                        feature=params["feature"]
                    )
                }

        elif graph_type == "Candles":
            params = {"ticker": ticker, "start_date": start_date_str, "end_date": end_date_str}
            key = cache_key("candles", params)
            st.session_state["eda_last"] = ("Candles", key)
            if key not in st.session_state:
                st.session_state[key] = {"df": load_day_df_for_candles(ticker, start_date_str, end_date_str)}

        elif graph_type == "Returns (with outliers)":
            params = {"ticker": ticker, "start_date": start_date_str, "end_date": end_date_str}
            key = cache_key("returns_outliers", params)
            st.session_state["eda_last"] = ("Returns (with outliers)", key)
            if key not in st.session_state:
                st.session_state[key] = {"df": load_day_df_for_returns(ticker, start_date_str, end_date_str)}

        elif graph_type == "Structural breaks (2 charts)":
            params = sb_params
            key = cache_key("sb2", params)
            st.session_state["eda_last"] = ("Structural breaks (2 charts)", key)
            if key not in st.session_state:
                st.session_state[key] = fetch_json("/data/structural-breaks", params=params)

        elif graph_type == "Correlation heatmap":
            params = {"ticker": ticker}
            key = cache_key("corr", params)
            st.session_state["eda_last"] = ("Correlation heatmap", key)
            if key not in st.session_state:
                st.session_state[key] = fetch_json("/data/correlation", params=params)

        elif graph_type == "ACF/PACF":
            params = acf_params
            key = cache_key("acf", params)
            st.session_state["eda_last"] = ("ACF/PACF", key)
            if key not in st.session_state:
                st.session_state[key] = fetch_json("/data/acf-pacf", params=params)

        elif graph_type == "Lag plot":
            params = lag_params
            key = cache_key("lag", params)
            st.session_state["eda_last"] = ("Lag plot", key)
            if key not in st.session_state:
                st.session_state[key] = fetch_json("/data/lag-plot", params=params)

    last = st.session_state.get("eda_last")
    if last:
        gtype, key = last
        data = st.session_state.get(key)

        with plot_area:
            if gtype == "Feature series":
                payload = data or {}
                feature_name = payload.get("feature")
                df = payload.get("df")

                if df is None or df.empty:
                    st.warning("Нет данных для выбранных параметров")
                else:
                    fig = go.Figure()
                    fig.add_trace(
                        go.Scatter(
                            x=df["Date"],
                            y=df["value"],
                            mode="lines",
                            name=feature_name
                        )
                    )

                    flag_col = _pick_outlier_flag_col(df)
                    if flag_col is not None and feature_name in ["Open", "High", "Low", "Close", "Volume"]:
                        out = df[df[flag_col] == 1]
                        if not out.empty:
                            fig.add_trace(
                                go.Scatter(
                                    x=out["Date"],
                                    y=out["value"],
                                    mode="markers",
                                    marker=dict(color="black", size=7),
                                    name="Выбросы"
                                )
                            )

                    fig.update_layout(
                        hovermode="x unified",
                        xaxis_title="Date",
                        yaxis_title=feature_name,
                        title=f"{ticker} — {feature_name}"
                    )
                    fig.update_xaxes(rangeslider_visible=True)
                    st.plotly_chart(fig, use_container_width=True)

            elif gtype == "Candles":
                df_c = data.get("df")
                if df_c is None or df_c.empty:
                    st.warning("Нет данных для свечей.")
                else:
                    fig = go.Figure()

                    fig.add_trace(
                        go.Candlestick(
                            x=df_c["Date"],
                            open=df_c["Open"],
                            high=df_c["High"],
                            low=df_c["Low"],
                            close=df_c["Close"],
                            name=f"{ticker} — Свечи",
                        )
                    )

                    flag_col = _pick_outlier_flag_col(df_c)
                    out = df_c[df_c[flag_col] == 1] if flag_col is not None else pd.DataFrame()
                    if not out.empty:
                        fig.add_trace(
                            go.Scatter(
                                x=out["Date"],
                                y=out["Close"],
                                mode="markers",
                                marker=dict(color="black", size=8, symbol="circle"),
                                name="Выбросы",
                            )
                        )

                    fig.update_layout(
                        title=f"{ticker} — свечи + выбросы",
                        hovermode="x unified",
                        xaxis_title="Дата",
                        yaxis_title="Цена",
                        xaxis_rangeslider_visible=False
                    )
                    st.plotly_chart(fig, use_container_width=True)

            elif gtype == "Returns (with outliers)":
                df = data["df"]
                if df is None or df.empty:
                    st.warning("Нет данных для выбранных параметров")
                else:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=df["Date"],
                        y=df["Return_pct"],
                        mode="lines",
                        name="Дневная доходность (%)"
                    ))

                    outliers = df[df["price_outlier"] == 1] if "price_outlier" in df.columns else pd.DataFrame()
                    if not outliers.empty:
                        fig.add_trace(go.Scatter(
                            x=outliers["Date"],
                            y=outliers["Return_pct"],
                            mode="markers",
                            name="Выбросы доходности",
                            marker=dict(color="red", size=8),
                        ))

                    fig.update_layout(
                        hovermode="x unified",
                        xaxis_title="Дата",
                        yaxis_title="Доходность (%)",
                        title=f"{ticker} — дневная доходность (%)"
                    )
                    fig.update_xaxes(rangeslider_visible=True)
                    st.plotly_chart(fig, use_container_width=True)

            elif gtype == "Structural breaks (2 charts)":
                df = pd.DataFrame(data["points"])
                df["date"] = pd.to_datetime(df["date"])
                df = df.sort_values("date")

                kernel_breaks = data.get("kernel_breaks", [])
                pelt_breaks = data.get("pelt_breaks", [])

                col_k, col_p = st.columns(2)

                with col_k:
                    fig_k = go.Figure()
                    for tr in _segment_traces(df, kernel_breaks, "date", "value", "Kernel"):
                        fig_k.add_trace(tr)
                    for d in kernel_breaks:
                        fig_k.add_vline(x=pd.to_datetime(d), line_width=1)
                    fig_k.update_layout(
                        title="Structural breaks — Kernel",
                        hovermode="x unified",
                        xaxis_title="Date",
                        yaxis_title="Close"
                    )
                    fig_k.update_xaxes(rangeslider_visible=True)
                    st.plotly_chart(fig_k, use_container_width=True)

                with col_p:
                    fig_p = go.Figure()
                    for tr in _segment_traces(df, pelt_breaks, "date", "value", "PELT"):
                        fig_p.add_trace(tr)
                    for d in pelt_breaks:
                        fig_p.add_vline(x=pd.to_datetime(d), line_width=1, line_dash="dash")
                    fig_p.update_layout(
                        title="Structural breaks — PELT",
                        hovermode="x unified",
                        xaxis_title="Date",
                        yaxis_title="Close"
                    )
                    fig_p.update_xaxes(rangeslider_visible=True)
                    st.plotly_chart(fig_p, use_container_width=True)

            elif gtype == "Correlation heatmap":
                feats = data["features"]
                mat = pd.DataFrame(data["matrix"], index=feats, columns=feats)
                fig = px.imshow(mat, aspect="auto")
                st.plotly_chart(fig, use_container_width=True)

            elif gtype == "ACF/PACF":
                lags = data["lags"]
                acf_vals = data["acf"]
                pacf_vals = data["pacf"]

                c1, c2 = st.columns(2)
                with c1:
                    fig1 = go.Figure()
                    fig1.add_trace(go.Bar(x=lags, y=acf_vals, name="ACF"))
                    fig1.update_layout(title="ACF", xaxis_title="Lag", yaxis_title="ACF")
                    st.plotly_chart(fig1, use_container_width=True)

                with c2:
                    fig2 = go.Figure()
                    fig2.add_trace(go.Bar(x=lags, y=pacf_vals, name="PACF"))
                    fig2.update_layout(title="PACF", xaxis_title="Lag", yaxis_title="PACF")
                    st.plotly_chart(fig2, use_container_width=True)

            elif gtype == "Lag plot":
                pts = pd.DataFrame(data["points"])
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=pts["x"], y=pts["y"],
                    mode="markers",
                    name=f'lag={data["lag"]}'
                ))
                fig.update_layout(
                    xaxis_title=f"{data['feature']}(t)",
                    yaxis_title=f"{data['feature']}(t+lag)"
                )
                st.plotly_chart(fig, use_container_width=True)
