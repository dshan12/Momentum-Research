import os
import sys

# Get the project root by going one level up from this file (i.e. src/)
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

# Prepend it to sys.path so Python will look here first
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


import pandas as pd
from src.signals import (
    load_prices,
    compute_monthly_returns,
    compute_momentum_signal,
    build_signals,
)


def backtest(longs, shorts, returns, tc=0.001):
    """
    Run an equal weighted long-short backtest with flat transaction costs.

    Parameters
    ----------
    longs: DataFrame[bool]
        True where we long at each date
    shorts: DataFrame[bool]
        True where we go short at each date
    returns: DataFrame[float]
        Monthly returns for each ticker
    tc: float
        Per-trade transaction costs, charged on entry and exit

    Returns
    -------
    strategy_rets: Series
        The monthly return series of the strategy
    """
    dates = returns.index.intersection(longs.index)
    strat_rets = list()

    # We will assume that there is a full turnover each month:
    # cost = tc * (n_longs + n_shorts) / universe_size * 2 (in+out)
    n_univ = returns.shape[1]

    for date in dates:
        # equal-weighted long return
        r_long = returns.loc[date, longs.loc[date]].mean()
        # equal-weighted short return
        r_short = returns.loc[date, shorts.loc[date]].mean()
        gross = r_long - r_short

        # approximate costs: both entry and exit on all positions
        n_positions = longs.loc[date].sum() + shorts.loc[date].sum()
        cost = tc * n_positions / n_univ * 2
        strat_rets.append(gross - cost)

    return pd.Series(strat_rets, index=dates)


if __name__ == "__main__":
    prices = load_prices()
    returns = compute_monthly_returns(prices)
    ranks = compute_momentum_signal(prices)
    longs, shorts = build_signals(ranks)

    # Backtest
    print("Running backtest...")
    strat = backtest(longs, shorts, returns)
    print(f"Strateg returns shape: {strat.shape}")
    print(strat.head())

    # Save to CSV for later analysis
    strat.to_csv("data/cleaned/strategy_monthly_returns.csv")
    print("Saved strategy returns to data/cleaned/strategy_monthly_returns.csv")
