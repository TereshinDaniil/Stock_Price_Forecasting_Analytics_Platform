# services/plots/outliers.py

import pandas as pd
import numpy as np


def detect_ohlcv_outliers(
    df: pd.DataFrame,
    price_z: float = 3.0,
    range_threshold: float = 20.0,
    volume_mad: float = 6.0,
    gap_threshold: float = 0.2
) -> pd.DataFrame:
    """
    Детектирует выбросы в OHLCV-данных и добавляет бинарные признаки:
      - price_outlier
      - range_outlier
      - volume_outlier
      - gap_outlier
      - any_outlier
    """

    df = df.copy()

    # --- единый контракт времени ---
    # в проекте мы используем Date (datetime64[ns, UTC])
    time_col = "Date"
    df[time_col] = pd.to_datetime(df[time_col], utc=True, errors="coerce")

    # сортировка по тикеру и времени
    df = df.sort_values(["Ticker", time_col]).reset_index(drop=True)

    # 1️⃣ выбросы по доходности
    df["Return"] = df.groupby("Ticker")["Close"].pct_change()
    mean_ret = df["Return"].mean()
    std_ret = df["Return"].std()

    df["price_outlier"] = (
        (df["Return"] > mean_ret + price_z * std_ret) |
        (df["Return"] < mean_ret - price_z * std_ret)
    ).astype(int)

    # 2️⃣ выбросы по диапазону свечи
    df["Range_pct"] = ((df["High"] - df["Low"]) / df["Low"]) * 100
    df["range_outlier"] = (df["Range_pct"] > range_threshold).astype(int)

    # 3️⃣ выбросы по объёму (MAD)
    grouped_vol = df.groupby("Ticker")["Volume"]
    median_vol = grouped_vol.transform("median")
    mad_vol = grouped_vol.transform(lambda x: (x - x.median()).abs().median())

    df["Volume_Z"] = 0.6745 * (df["Volume"] - median_vol) / mad_vol
    df["volume_outlier"] = (df["Volume_Z"].abs() > volume_mad).astype(int)

    # 4️⃣ выбросы по гэпам (Open к предыдущему Close)
    df["Gap"] = df.groupby("Ticker")["Open"].pct_change()
    df["gap_outlier"] = (df["Gap"].abs() > gap_threshold).astype(int)

    # 5️⃣ итоговый флаг
    df["any_outlier"] = (
        df["price_outlier"] |
        df["range_outlier"] |
        df["volume_outlier"] |
        df["gap_outlier"]
    ).astype(int)

    return df
