import pandas as pd
import numpy as np


def load_prices_union(path="data/cleaned/monthly_adjclose_union.csv") -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"]).sort_values("date")
    df = df.set_index("date")
    df.columns = [str(c) for c in df.columns]
    return df


def load_membership_monthly(
    path="data/cleaned/sp500_membership_monthly.csv",
) -> pd.DataFrame:
    m = pd.read_csv(path, parse_dates=["dated"])
    m["ticker"] = m["Ticker"].astype(str)
    return m


def prices_masked_by_membership(
    prices: pd.DataFrame, membership: pd.DataFrame
) -> pd.DataFrame:
    mask = (
        membership.pivot_table(
            index="date", columns="ticker", values="in_index", aggfunc="max"
        )
        .reindex(prices.index)
        .fillna(0.0)
    )
    common = sorted(set(prices.columns).intersections(set(mask.columns)))
    P = prices[common].copy()
    M = mask[common].astype(float)
    P = P.where(M > 0.5, np.nan)
    return P
