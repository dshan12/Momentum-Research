import os
import numpy as np
import pandas as pd

from data.panel_from_membership import (
    load_prices_union,
    load_membership_monthly,
    prices_masked_by_membership,
)


def compute_monthly_returns_from_masked_prices(
    masked_prices: pd.DataFrame,
) -> pd.DataFrame:
    rets = masked_prices.pct_change(fill_method=None)
    logr = np.log1p(rets)
    bad = logr.abs() > 1.5
    rets = rets.mask(bad)

    def _winsorize_row(row):
        if row.notna().sum() < 50:
            return row
        lo, hi = row.quantile(0.01), row.quantile(0.99)
        return row.clip(lower=lo, upper=hi)

    rets = rets.apply(_winsorize_row, axis=1)

    return rets


def compute_momentum_ranks(
    masked_prices: pd.DataFrame, lookback=12, skip=1
) -> pd.DataFrame:
    L = np.log(masked_prices)
    mom = (L.shift(skip) - L.shift(lookback + skip)).replace([np.inf, -np.inf], np.nan)
    ranks = mom.rank(axis=1, pct=True, na_option="keep")
    return ranks


def build_signals(ranks: pd.DataFrame, long_q=0.9, short_q=0.1):
    longs = (ranks >= long_q).astype(int)
    shorts = (ranks <= short_q).astype(int)
    return longs, shorts


def long_short_equal_weight(
    longs: pd.DataFrame, shorts: pd.DataFrame, rets: pd.DataFrame
) -> pd.Series:
    L = longs.astype(float)
    S = shorts.astype(float)
    lw = L.div(L.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    sw = -S.div(S.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    W = lw.add(sw, fill_value=0.0)
    strat = (W * rets).sum(axis=1)
    strat.name = "strategy"
    return strat


def annualized_return(r: pd.Series) -> float:
    return (1 + r.mean()) ** 12 - 1


def sharpe_ratio(r: pd.Series, rf_annual: float = 0.02) -> float:
    rf_m = (1 + rf_annual) ** (1 / 12) - 1
    ex = r - rf_m
    return (ex.mean() / (ex.std() + 1e-12)) * np.sqrt(12)


def max_drawdown(r: pd.Series) -> float:
    w = (1 + r).cumprod()
    peak = w.cummax()
    dd = (w / peak) - 1
    return float(dd.min())


def main():
    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

    prices = load_prices_union(
        os.path.join(ROOT, "data/cleaned/monthly_adjclose_union.csv")
    )
    prices = prices.where(prices > 0)
    membership = load_membership_monthly(
        os.path.join(ROOT, "data/cleaned/sp500_membership_monthly.csv")
    )

    masked_prices = prices_masked_by_membership(prices, membership)
    rets = compute_monthly_returns_from_masked_prices(masked_prices)
    # rets = rets.mask((rets > 3.0) | (rets < -0.9))
    ranks = compute_momentum_ranks(masked_prices, lookback=12, skip=1)
    longs, shorts = build_signals(ranks, long_q=0.9, short_q=0.1)

    min_names = 20
    valid = (longs.sum(axis=1) >= min_names) & (shorts.sum(axis=1) >= min_names)
    longs = longs.where(valid, 0)
    shorts = shorts.where(valid, 0)
    gross = long_short_equal_weight(longs, shorts, rets)

    TC = 0.001
    names_traded = longs.sum(axis=1).fillna(0) + shorts.sum(axis=1).fillna(0)
    ntickers = (masked_prices.notna().sum(axis=1)).astype(float).replace(0, np.nan)
    cost = (2 * TC * names_traded / ntickers).fillna(0.0)

    net = (gross - cost).dropna()

    print("=== Survivorship-free (12–1, name-count costs 10 bps) ===")
    print(f"Ann Return (net): {annualized_return(net):.2%}")
    print(
        f"Vol/Sharpe (2% rf): {(net.std() * np.sqrt(12)):.2%} / {sharpe_ratio(net):.2f}"
    )
    print(f"Max DD: {max_drawdown(net):.2%}")

    out_dir = os.path.join(ROOT, "data/cleaned")
    os.makedirs(out_dir, exist_ok=True)
    masked_prices.to_csv(os.path.join(out_dir, "masked_prices_survivorship.csv"))
    rets.to_csv(os.path.join(out_dir, "returns_survivorship.csv"))
    gross.to_csv(os.path.join(out_dir, "strategy_gross_survivorship.csv"))
    net.to_csv(os.path.join(out_dir, "strategy_net_survivorship.csv"))
    print("[✓] Saved survivorship-free series.")


if __name__ == "__main__":
    main()
