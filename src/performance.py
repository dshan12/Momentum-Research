import numpy as np
import pandas as pd
import statsmodels.api as sm


def annualized_returns(returns):
    """
    Compute annualized returns from a monthly-return series.
    """
    compounded = (1 + returns).prod()
    years = len(returns) / 12
    return compounded ** (1 / years) - 1


def annualized_volatility(returns):
    """
    Compute annualized volatility from monthly returns.
    """
    return returns.std() * np.sqrt(12)


def sharpe_ratio(returns, rf_rate=0.0):
    """
    Compute annualized Sharpe ratio (rf_rate is an annual number).
    """
    # convert annual RF to monthly rate
    rf_m = (1 + rf_rate) ** (1 / 12) - 1
    excess = returns - rf_m
    ann_ex_ret = annualized_returns(excess)
    ann_vol = annualized_volatility(returns)
    return ann_ex_ret / ann_vol


def max_drawdown(returns):
    """
    Compute maximum drawdown (as a positive fraction) from monthly returns.
    """
    wealth = (1 + returns).cumprod()
    peak = wealth.cummax()
    drawdown = (peak - wealth) / peak
    return drawdown.max()


def capm_regression(returns, market_returns, rf_rate=0.0):
    """
    Run CAPM regression:
      (R_p - R_f) = α + β (R_m - R_f) + ε
    Returns a **fitted** statsmodels RegressionResults object.
    """
    # align indices
    common = returns.index.intersection(market_returns.index)
    r = returns.loc[common]
    m = market_returns.loc[common]

    rf_m = (1 + rf_rate) ** (1 / 12) - 1
    y = r - rf_m
    x = m - rf_m
    X = sm.add_constant(x)

    results = sm.OLS(y, X).fit()
    return results


def bootstrap_alpha(returns, market_returns, rf_rate=0.0, n_iters=1000, seed=None):
    """
    Bootstrap test on CAPM alpha:
      - Aligns inputs
      - Computes original α
      - Resamples months with replacement n_iters times
      - Re-fits CAPM each time, collects α_i
      - Returns (orig_alpha, two-sided p-value)
    """
    # 1. Align
    common = returns.index.intersection(market_returns.index)
    r = returns.loc[common]
    m = market_returns.loc[common]

    # 2. Original α
    orig_res = capm_regression(r, m, rf_rate=rf_rate)
    orig_alpha = orig_res.params["const"]

    # 3. Seed
    if seed is not None:
        np.random.seed(seed)

    # 4. Bootstrap loop
    data = pd.DataFrame({"asset": r, "market": m})
    alphas = []
    for _ in range(n_iters):
        sample = data.sample(frac=1.0, replace=True)
        res_i = capm_regression(sample["asset"], sample["market"], rf_rate=rf_rate)
        alphas.append(res_i.params["const"])

    alphas = np.array(alphas)

    # 5. Empirical two-sided p-value
    p_val = np.mean(np.abs(alphas) >= abs(orig_alpha))
    return orig_alpha, p_val


if __name__ == "__main__":
    # Demo of metrics – adjust these imports if you run from project root
    from backtest import backtest
    from signals import (
        load_prices,
        compute_monthly_returns,
        compute_momentum_signal,
        build_signals,
    )
    import yfinance as yf

    # 1. Build the strategy returns
    prices = load_prices()
    rets = compute_monthly_returns(prices)
    ranks = compute_momentum_signal(prices)
    longs, shorts = build_signals(ranks)
    strat = backtest(longs, shorts, rets)

    # 2. Print basic stats
    print("Ann. Return    ", f"{annualized_returns(strat):.2%}")
    print("Ann. Volatility", f"{annualized_volatility(strat):.2%}")
    print("Sharpe (2% rf) ", f"{sharpe_ratio(strat, rf_rate=0.02):.2f}")
    print("Max Drawdown  ", f"{max_drawdown(strat):.2%}")

    # 3. CAPM on strat vs. SPY
    spy = yf.download("SPY", start="2005-01-01", end="2024-12-31", interval="1mo")[
        "Close"
    ]
    sp_rets = spy.pct_change().dropna().iloc[:, 0]
    capm_res = capm_regression(strat, sp_rets, rf_rate=0.02)
    print(capm_res.summary())

    # 4. Bootstrap
    alpha, pval = bootstrap_alpha(strat, sp_rets, rf_rate=0.02, n_iters=1000, seed=42)
    print(f"Bootstrap α: {alpha:.4%}, p-value: {pval:.3f}")
