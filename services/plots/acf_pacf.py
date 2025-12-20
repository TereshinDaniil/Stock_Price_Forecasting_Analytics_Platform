import io
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import warnings

warnings.filterwarnings("ignore")


def plot_acf_pacf(
    ticker: str,
    feature: str,
    lags: int = 40,
    stationarity_df: pd.DataFrame | None = None,
):
    """
    ACF / PACF по СЫРОМУ ряду за ВЕСЬ период
    + аннотация стационарности (если передана).
    """

    # 1️⃣ загрузка дневных данных
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

    s = df[feature].dropna()

    if len(s) < max(10, lags + 5):
        raise ValueError("Недостаточно наблюдений для ACF / PACF")

    # 2️⃣ заголовок + стационарность
    title = f"{ticker} — {feature}"

    if (
        stationarity_df is not None
        and {"Ticker", "Feature", "IsStationary", "Test", "PValue"}
        <= set(stationarity_df.columns)
    ):
        row = stationarity_df[
            (stationarity_df["Ticker"] == ticker) &
            (stationarity_df["Feature"] == feature)
        ]

        if not row.empty:
            status = (
                "Стационарен"
                if int(row["IsStationary"].iloc[0]) == 1
                else "Нестационарен"
            )
            test = row["Test"].iloc[0]
            pval = float(row["PValue"].iloc[0])
            title += f" | {status} ({test}, p={pval:.3g})"

    # 3️⃣ графики
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    plot_acf(s, lags=lags, ax=axes[0])
    axes[0].set_title(f"ACF — {ticker} ({feature})")
    axes[0].grid(True, alpha=0.3)

    plot_pacf(s, lags=lags, ax=axes[1], method="ywmle")
    axes[1].set_title(f"PACF — {ticker} ({feature})")
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14)

    # 4️⃣ buffer для FastAPI
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)

    return buf
