import os
import sys

# Get the project root by going one level up from this file (i.e. src/)
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

# Prepend it to sys.path so Python will look here first
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


import pandas as pd
import sys
import os

# Compute the absolute path of your project root (one level up)
root = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.insert(0, root)


from src.signals import (
    load_prices,
    compute_monthly_returns,
    compute_momentum_signal,
    build_signals,
)
from src.backtest import backtest
from src.performance import (
    annualized_returns,
    sharpe_ratio,
    max_drawdown,
)


def vary_lookbacks(prices, lookbacks=[6, 9, 12, 18], skip=1, tc=0.001):
    """
    Run backtests on a range of lookbacks, and return a dataframe of results.
    """
    rets = compute_monthly_returns(prices)
    rows = list()
    for lb in lookbacks:
        ranks = compute_momentum_signal(prices, lookback=lb, skip=skip)
        longs, shorts = build_signals(ranks)
        strat = backtest(longs, shorts, rets, tc=tc)
        rows.append(
            {
                "lookback": lb,
                "ann_return": annualized_returns(strat),
                "sharpe": sharpe_ratio(strat, rf_rate=0.02),
                "max_drawdown": max_drawdown(strat),
            }
        )
    return pd.DataFrame(rows).set_index("lookback")


def vary_tc(prices, lookback=12, skip=1, tcs=(0, 0.001, 0.002, 0.005)):
    """Run backtests for several transaction-cost levels"""
    rets = compute_monthly_returns(prices)
    ranks = compute_momentum_signal(prices, lookback=lookback, skip=skip)
    longs, shorts = build_signals(ranks)
    rows = list()
    for tc in tcs:
        strat = backtest(longs, shorts, rets, tc=tc)
        rows.append(
            {
                "tc_bps": int(tc * 10000),
                "ann_return": annualized_returns(strat),
                "sharpe": sharpe_ratio(strat, rf_rate=0.02),
                "max_drawdown": max_drawdown(strat),
            }
        )
    return pd.DataFrame(rows).set_index("tc_bps")


def fetch_sector_map():
    """
    Returns a dict mapping ticker → sector using Wikipedia's S&P 500 list.
    """
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    table = pd.read_html(url, header=0)[0]
    table["Symbol"] = table["Symbol"].str.replace(".", "-", regex=False)
    return dict(zip(table["Symbol"], table["GICS Sector"]))


def sector_neutral(prices, sector_map, lookback=12, skip=1, tc=0.001):
    """
    Rank within each sector, then equal weight across sectors to neutralize sector tilts
    """
    rets = compute_monthly_returns(prices)
    mom = prices.pct_change(lookback).shift(skip).dropna(how="all")
    sectors = pd.Series(sector_map).reindex(prices.columns)
    rank_df = mom.groupby(sectors, axis=1).rank(axis=1, pct=True)
    longs, shorts = build_signals(rank_df)
    return backtest(longs, shorts, rets, tc=tc)


if __name__ == "__main__":
    p = load_prices()

    print("=== Lookback Sensitivity ===")
    print(vary_lookbacks(p).round(4))

    print("\n=== Transaction Cost Sensitivity ===")
    print(vary_tc(p).round(4))

    print("\n=== Sector-Neutral Strategy ===")
    sector_map = fetch_sector_map()
    sn_rets = sector_neutral(p, sector_map)
    print(f"Ann. Return: {annualized_returns(sn_rets):.2%}")
    print(f"Sharpe (2% rf): {sharpe_ratio(sn_rets, rf_rate=0.02):.2f}")
    print(f"Max Drawdown: {max_drawdown(sn_rets):.2%}")
