import matplotlib
matplotlib.use("Agg")

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from io import BytesIO


DATA_PATH = Path("data/day_data.parquet")


def plot_close_price(
    ticker: str,
    start_date: str,
    end_date: str,
) -> BytesIO:
    # --- load data ---
    df = pd.read_parquet(DATA_PATH)

    df["Date"] = pd.to_datetime(df["Date"], utc=True)

    # --- filter ---
    mask = (
        (df["Ticker"] == ticker)
        & (df["Date"] >= pd.to_datetime(start_date, utc=True))
        & (df["Date"] <= pd.to_datetime(end_date, utc=True))
    )

    data = df.loc[mask, ["Date", "Close"]].sort_values("Date")

    if data.empty:
        raise ValueError("No data for given parameters")

    # --- plot ---
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(data["Date"], data["Close"], linewidth=2)
    ax.set_title(f"{ticker} — Close price", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price ($)")
    ax.grid(alpha=0.3)

    for label in ax.get_xticklabels():
        label.set_rotation(30)
        label.set_horizontalalignment("right")

    # --- save to buffer ---
    buf = BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)

    return buf
