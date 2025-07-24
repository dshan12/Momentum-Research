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
        strat = backtest(longs, shorts, rets)
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


def sector_neutral(prices, sector_map, lookback=12, skip=1, tc=0.001):
    """
    Rank within each sector, then equal weight across sectors to neutralize sector tilts
    """
    rets = compute_monthly_returns(prices)
    mom = prices.pct_chage(lookback).shift(skip).dropna(how="all")
    sectors = pd.Series(sector_map).reindex(prices.columns)
    rank_df = (mom.groupby(sectors, axis=1).mean() * 100).rank(axis=1, pct=True)
    longs, shorts = build_signals(rank_df)
    return backtest(longs, shorts, rets, tc=tc)


if __name__ == "__main__":
    p = load_prices()
    print(vary_lookbacks(p))
    print(vary_tc(p))
