import pandas as pd
import numpy as np


def normalize_weights(W: pd.DataFrame) -> pd.DataFrame:
    row_sum = W.abs().sum(axis=1).replace(0, np.nan)
    return W.div(row_sum, axis=0).fillna(0.0)


def equal_weight_long_short(longs: pd.DataFrame, shorts: pd.DataFrame) -> pd.DataFrame:
    L = (longs > 0).astype(float)
    S = (shorts > 0).astype(float)
    lw = L.div(L.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    sw = -S.div(S.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    W = lw.add(sw, fill_value=0.0)
    return normalize_weights(W)


def drift_weights(W_prev: pd.Series, rets_prev: pd.Series) -> pd.Series:
    if W_prev is None or W_prev.empty:
        return W_prev
    aligned = pd.concat([W_prev, rets_prev], axis=1, keys=["w", "r"]).dropna(how="all")
    if aligned.empty:
        return W_prev
    w = aligned["w"].fillna(0.0)
    r = aligned["r"].fillna(0.0)
    w_post = w * (1.0 + r)
    denom = w_post.abs().sum()
    if denom == 0 or pd.isna(denom):
        return w_post
    return w_post / denom


def turnover_from_weights(W: pd.DataFrame, rets: pd.DataFrame) -> pd.Series:
    to = []
    dates = W.index
    for i, t in enumerate(dates):
        if i == 0:
            to.append(0.0)
            continue
        W_prev = W.iloc[i - 1]
        rets_prev = rets.iloc[i - 1] if rets is not None else None
        w_pre = drift_weights(W_prev, rets_prev) if rets_prev is not None else W_prev
        w_t = W.iloc[i].reindex(W.columns).fillna(0.0)
        w_pre = w_pre.reindex(W.columns).fillna(0.0)
        to.append(0.5 * (w_t - w_pre).abs().sum())
    s = pd.Series(to, index=dates, name="turnover")
    return s


def apply_turnover_costs(
    gross: pd.Series, turnover: pd.Series, cost_bps: float
) -> pd.Series:
    c = cost_bps / 10000
    net = gross - turnover * c
    net.name = "strategy_net"
    return net
