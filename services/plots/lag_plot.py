import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")


def plot_lag_plot(
    ticker: str,
    feature: str,
    lag: int = 1,
):
    """
    Lag plot по дневным данным (весь период).
    """

    # 1️⃣ загрузка данных
    df = pd.read_parquet("data/day_data.parquet")
    df["Date"] = pd.to_datetime(df["Date"], utc=True, errors="coerce")

    df = (
        df[df["Ticker"] == ticker]
        .sort_values("Date")
        .reset_index(drop=True)
    )

    if df.empty:
        raise ValueError("Нет данных для выбранного тикера")

    if feature not in df.columns:
        raise ValueError(f"Признак '{feature}' не найден")

    series = df[feature].dropna()

    if len(series) <= lag + 5:
        raise ValueError("Недостаточно данных для выбранного лага")

    # 2️⃣ формирование lagged df
    df_lag = pd.DataFrame({
        "y": series
    })
    df_lag["lagged"] = df_lag["y"].shift(lag)
    df_lag = df_lag.dropna()

    # 3️⃣ построение графика
    fig, ax = plt.subplots(figsize=(6, 6))

    ax.scatter(
        df_lag["lagged"],
        df_lag["y"],
        s=8,
        alpha=0.5,
        color="tab:blue",
        edgecolors="none"
    )

    # регрессионная линия
    if len(df_lag) > 2:
        x = df_lag["lagged"].values
        y = df_lag["y"].values
        A = np.vstack([x, np.ones_like(x)]).T
        a, b = np.linalg.lstsq(A, y, rcond=None)[0]
        xs = np.linspace(x.min(), x.max(), 100)
        ax.plot(xs, a * xs + b, color="red", linewidth=1.5)

    ax.set_title(f"{ticker} — {feature} | Lag plot (lag={lag})")
    ax.set_xlabel(f"{feature}(t-{lag})")
    ax.set_ylabel(f"{feature}(t)")
    ax.grid(True, alpha=0.3)

    # 4️⃣ buffer
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)

    return buf
