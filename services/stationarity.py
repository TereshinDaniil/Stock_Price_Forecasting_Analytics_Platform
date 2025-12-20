import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss


def check_stationarity(
    df: pd.DataFrame,
    target_cols: list[str] | None = None,
    test: str = "ADF",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Проверка стационарности временных рядов (Ticker × Feature).

    Возвращает DataFrame:
    Ticker | Feature | Test | TestStatistic | PValue | IsStationary
    """

    time_col = "Datetime" if "Datetime" in df.columns else "Date"
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col], utc=True, errors="coerce")

    # если признаки не заданы — берём все числовые
    if not target_cols:
        exclude = {time_col, "Ticker"}
        target_cols = [
            c for c in df.columns
            if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
        ]

    results = []

    for ticker, group in df.groupby("Ticker"):
        group = group.sort_values(time_col)

        for feature in target_cols:
            series = group[feature].dropna()

            if len(series) < 10:
                continue

            if test.upper() == "ADF":
                stat, pvalue, *_ = adfuller(series)
                is_stationary = int(pvalue < alpha)

            elif test.upper() == "KPSS":
                stat, pvalue, *_ = kpss(series, regression="c", nlags="auto")
                is_stationary = int(pvalue > alpha)

            else:
                raise ValueError("test должен быть 'ADF' или 'KPSS'")

            results.append({
                "Ticker": ticker,
                "Feature": feature,
                "Test": test.upper(),
                "TestStatistic": stat,
                "PValue": pvalue,
                "IsStationary": is_stationary,
            })

    return pd.DataFrame(results)
