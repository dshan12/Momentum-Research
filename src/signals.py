import pandas as pd


def load_prices(path="data/cleaned/cleaned_monthly_prices.csv"):
    """Loads the cleaned monthly prices csv file"""
    prices = pd.read_csv(path, index_col=0, parse_dates=True)
    return prices


def compute_montly_returns(prices):
    """Compute simple montly returns from price series"""
    return prices.pct_change().dropna(how="all")


def compute_momentum_signal(prices, lookback=12, skip=1):
    """
    Compute cross-secitonal momentum:
    - Lookback-month total returns
    - Skip the most recent month
    - rank into percentiles
    Returns a dataframe of percentile ranks [0-1]
    """
    # Total return over the past 'lookback' months, shiften by 'skip' month
    mom = prices.pct_change(lookback).shift(skip).dropna(how="all")
    # Cross-secitonal rank [0-1]
    ranks = mom.rank(axis=1, pct=True)
    return ranks


def build_signals(ranks, long_pct=0.1, short_pct=0.1):
    """
    From percentile raks, build boolean long/short signals
    - long: top 'long_pct' (>=1-long_pct)
    - short: bottom 'short_pct' (<=short_pct)
    """
    longs = ranks >= (1 - long_pct)
    shorts = ranks <= short_pct
    return longs, shorts


if __name__ == "__main__":
    # Quick standalone test
    print("Loading prices...")
    prices = load_prices()
    print("Computing montly returns...")
    returns = compute_montly_returns(prices)
    print("Computing momentum signal...")
    ranks = compute_momentum_signal(returns)
    print("Building signals...")
    longs, shorts = build_signals(ranks)
    print("Signals built!")
    print(f"Returns shape: {returns.shape}")
    print(f"Signal shape: {ranks.shape}, longs={longs.sum().sum()}")
