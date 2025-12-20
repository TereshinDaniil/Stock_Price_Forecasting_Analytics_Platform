import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from services.structural_breaks import (
    detect_kernel_changepoints,
    detect_pelt_changepoints,
)


def plot_structural_breaks(
    ticker: str,
    start_date: str,
    end_date: str,
    kernel_n_bkps: int = 5,
    pelt_penalty: int = 5,
):
    # --- данные ---
    df = pd.read_parquet("data/day_data.parquet")
    df["Date"] = pd.to_datetime(df["Date"], utc=True, errors="coerce")

    df = df[
        (df["Ticker"] == ticker) &
        (df["Date"] >= pd.to_datetime(start_date, utc=True)) &
        (df["Date"] <= pd.to_datetime(end_date, utc=True))
    ]

    if df.empty:
        raise ValueError("Нет данных для выбранных параметров")

    # --- KernelCPD ---
    kernel_df = detect_kernel_changepoints(
        df=df,
        target_cols=("Close",),
        kernel="rbf",
        n_bkps=kernel_n_bkps,
        min_size=30,
    )

    kernel_df = kernel_df[kernel_df["Ticker"] == ticker]

    # --- PELT ---
    pelt_df = detect_pelt_changepoints(
        df=df,
        target_cols=("Close",),
        cost_model="rbf",
        penalty=pelt_penalty,
        min_size=30,
    )

    pelt_df = pelt_df[pelt_df["Ticker"] == ticker]

    # --- график ---
    fig, axes = plt.subplots(2, 1, figsize=(14, 12), sharex=True)

    # KernelCPD
    ax = axes[0]
    for seg_id, seg_df in kernel_df.groupby("segment_id"):
        ax.plot(seg_df["Date"], seg_df["Close"], linewidth=2)
    ax.scatter(
        kernel_df[kernel_df["changepoint"] == 1]["Date"],
        kernel_df[kernel_df["changepoint"] == 1]["Close"],
        color="red",
        s=50,
        label="Changepoints",
    )
    ax.set_title(f"{ticker} — KernelCPD (n_bkps={kernel_n_bkps})")
    ax.grid(alpha=0.3)

    # PELT
    ax = axes[1]
    for seg_id, seg_df in pelt_df.groupby("segment_id"):
        ax.plot(seg_df["Date"], seg_df["Close"], linewidth=2)
    ax.scatter(
        pelt_df[pelt_df["changepoint"] == 1]["Date"],
        pelt_df[pelt_df["changepoint"] == 1]["Close"],
        color="red",
        s=50,
        label="Changepoints",
    )
    ax.set_title(f"{ticker} — PELT (penalty={pelt_penalty})")
    ax.grid(alpha=0.3)

    axes[1].set_xlabel("Date")
    axes[0].set_ylabel("Close price")
    axes[1].set_ylabel("Close price")

    import io
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)

    return buf
