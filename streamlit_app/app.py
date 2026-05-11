from __future__ import annotations

from datetime import date, timedelta

import streamlit as st

from utils import get_features, get_ticker_date_range, get_tickers


st.set_page_config(page_title="Time Series Pipeline", layout="wide")

st.title("Time Series Pipeline")
st.caption("Выберите данные, затем переходите по шагам: EDA -> Fit model -> Future forecast.")

tickers = get_tickers()
features = get_features()

if not tickers:
    st.error("Список тикеров пуст.")
    st.stop()

left, right = st.columns([1, 2])


def _stored_date(key: str, fallback: date) -> date:
    value = st.session_state.get(key, fallback)
    if isinstance(value, str):
        return date.fromisoformat(value)
    return value

with left:
    ticker = st.selectbox(
        "Ticker",
        tickers,
        index=tickers.index(st.session_state["ticker"]) if st.session_state.get("ticker") in tickers else 0,
    )

    min_date, max_date = get_ticker_date_range(ticker)
    default_end = _stored_date("end_date", max_date)
    default_start = _stored_date("start_date", max(min_date, max_date - timedelta(days=365)))

    start_date = st.date_input(
        "Start date",
        value=max(min_date, min(default_start, max_date)),
        min_value=min_date,
        max_value=max_date,
    )
    end_date = st.date_input(
        "End date",
        value=max(min_date, min(default_end, max_date)),
        min_value=min_date,
        max_value=max_date,
    )

    target_default = st.session_state.get("target", "Close")
    target = st.selectbox(
        "Target",
        features,
        index=features.index(target_default) if target_default in features else 0,
    )

    if start_date > end_date:
        st.error("Start date must be before end date.")
        st.stop()

    if st.button("Сохранить выбор", type="primary"):
        st.session_state["ticker"] = ticker
        st.session_state["target"] = target
        st.session_state["start_date"] = start_date.strftime("%Y-%m-%d")
        st.session_state["end_date"] = end_date.strftime("%Y-%m-%d")
        st.success("Выбор сохранён.")

with right:
    st.subheader("Текущий контекст")
    if "ticker" in st.session_state:
        st.json(
            {
                "ticker": st.session_state["ticker"],
                "target": st.session_state["target"],
                "start_date": st.session_state["start_date"],
                "end_date": st.session_state["end_date"],
            }
        )
    else:
        st.info("Сохраните выбор слева, чтобы открыть следующие шаги.")

    st.page_link("pages/1_EDA.py", label="1. EDA", disabled="ticker" not in st.session_state)
    st.page_link("pages/2_Model_Fit.py", label="2. Fit model", disabled="ticker" not in st.session_state)
    st.page_link("pages/3_Forecast.py", label="3. Future forecast", disabled="ticker" not in st.session_state)
