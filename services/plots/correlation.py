import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


def plot_feature_correlation(
    ticker: str,
):
    # 1) загружаем дневные данные
    df = pd.read_parquet("data/day_data.parquet")
    df["Date"] = pd.to_datetime(df["Date"], utc=True, errors="coerce")

    # 2) фильтрация по тикеру
    df = df[df["Ticker"] == ticker].copy()

    if df.empty:
        raise ValueError("Нет данных для выбранного тикера")

    # 3) только числовые признаки
    numeric_df = df.select_dtypes(include="number")

    if numeric_df.shape[1] < 2:
        raise ValueError("Недостаточно числовых признаков для корреляции")

    # 4) корреляционная матрица
    corr_matrix = numeric_df.corr()

    # 5) визуализация
    plt.figure(figsize=(10, 8))
    sns.set(style="whitegrid")

    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".6f",
        cmap="coolwarm",
        linewidths=0.5,
        cbar_kws={"label": "Корреляция"},
        square=True
    )

    plt.title(
        f"{ticker} — корреляционная матрица признаков",
        fontsize=14,
        pad=12
    )
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    import io
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", dpi=150)
    plt.close()
    buf.seek(0)

    return buf
