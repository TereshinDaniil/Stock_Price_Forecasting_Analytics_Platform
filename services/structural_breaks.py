import pandas as pd
import numpy as np
import ruptures as rpt


def detect_kernel_changepoints(
    df: pd.DataFrame,
    target_cols=("Close",),
    kernel="rbf",
    n_bkps=5,
    min_size=30,
) -> pd.DataFrame:

    results = []

    for ticker, group in df.groupby("Ticker"):
        group = group.sort_values("Date").reset_index(drop=True)
        X = group[list(target_cols)].values

        model = rpt.KernelCPD(kernel=kernel, min_size=min_size).fit(X)
        bkps = model.predict(n_bkps=n_bkps)

        group["changepoint"] = 0
        for bkp in bkps[:-1]:
            group.loc[bkp, "changepoint"] = 1

        segment_ids = np.zeros(len(group), dtype=int)
        current_segment = 0
        last_bkp = 0
        for bkp in bkps:
            segment_ids[last_bkp:bkp] = current_segment
            current_segment += 1
            last_bkp = bkp

        group["segment_id"] = segment_ids
        results.append(group)

    return pd.concat(results, ignore_index=True)


def detect_pelt_changepoints(
    df: pd.DataFrame,
    target_cols=("Close",),
    cost_model="rbf",
    penalty=5,
    min_size=30,
) -> pd.DataFrame:
    results = []

    for ticker, group in df.groupby("Ticker"):
        group = group.sort_values("Date").reset_index(drop=True)
        X = group[list(target_cols)].values

        model = rpt.Pelt(model=cost_model, min_size=min_size).fit(X)
        bkps = model.predict(pen=penalty)

        group["changepoint"] = 0
        for bkp in bkps[:-1]:
            group.loc[bkp, "changepoint"] = 1

        segment_ids = np.zeros(len(group), dtype=int)
        current_segment = 0
        last_bkp = 0
        for bkp in bkps:
            segment_ids[last_bkp:bkp] = current_segment
            current_segment += 1
            last_bkp = bkp

        group["segment_id"] = segment_ids
        results.append(group)

    return pd.concat(results, ignore_index=True)

