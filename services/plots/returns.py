import matplotlib.pyplot as plt
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from services.outliers import detect_ohlcv_outliers


def plot_daily_returns(
    ticker: str,
    start_date: str,
    end_date: str,
):
    # 1) читаем данные (КАК В close_price)
    df = pd.read_parquet("data/day_data.parquet")
    df["Date"] = pd.to_datetime(df["Date"], utc=True, errors="coerce")

    # 2) фильтрация
    df = df[
        (df["Ticker"] == ticker) &
        (df["Date"] >= pd.to_datetime(start_date, utc=True)) &
        (df["Date"] <= pd.to_datetime(end_date, utc=True))
    ]

    if df.empty:
        raise ValueError("Нет данных для выбранных параметров")

    # 3) ровно как в pandas
    df = detect_ohlcv_outliers(df)
    df["Return_pct"] = df["Return"] * 100
    outliers = df[df["price_outlier"] == 1]

    # 4) график
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(
        df["Date"],
        df["Return_pct"],
        linewidth=2.0,
        label="Дневная доходность (%)"
    )

    ax.scatter(
        outliers["Date"],
        outliers["Return_pct"],
        color="red",
        s=40,
        label="Выбросы доходности",
        zorder=3
    )

    ax.set_title(
        f"{ticker} — дневная доходность (%)",
        fontsize=14,
        fontweight="bold"
    )
    ax.set_xlabel("Дата")
    ax.set_ylabel("Доходность (%)")
    ax.legend()
    ax.grid(alpha=0.3)

    import io
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)

    return buf
