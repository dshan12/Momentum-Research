import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
DATA_DIR = os.path.join(ROOT, "data", "cleaned")
FIG_DIR = os.path.join(ROOT, "figures")

NET_PATH = os.path.join(DATA_DIR, "strategy_net_survivorship.csv")
FF_PATH = os.path.join(DATA_DIR, "ff5_umd_monthly.csv")


def load_series(path: str, name: str) -> pd.Series:
    s = pd.read_csv(path, parse_dates=[0], index_col=0).squeeze("columns")
    if isinstance(s, pd.DataFrame):
        assert s.shape[1] == 1, f"Expected 1 column in {path}"
        s = s.iloc[:, 0]
    s.name = name
    return s.sort_index().asfreq("ME")


def build_us_market(ff: pd.DataFrame) -> pd.Series:
    return (ff["Mkt-RF"] + ff["RF"]).rename("US Market").asfreq("ME")


def cumulative_wealth(r: pd.Series, start_value: float = 1.0) -> pd.Series:
    return start_value * (1.0 + r.fillna(0.0)).cumprod()


def drawdown_series(r: pd.Series) -> pd.Series:
    w = cumulative_wealth(r, 1.0)
    peak = w.cummax()
    dd = (w / peak) - 1.0
    dd.name = "Drawdown"
    return dd


def rolling_sharpe_excess(
    ret: pd.Series, rf: pd.Series, window: int = 36, min_periods: int = 12
) -> pd.Series:
    ex = (ret - rf).dropna()
    mu = ex.rolling(window, min_periods=min_periods).mean()
    sd = ex.rolling(window, min_periods=min_periods).std()
    sr = (mu / (sd + 1e-12)) * np.sqrt(12.0)
    return sr


def main():
    os.makedirs(FIG_DIR, exist_ok=True)

    net = load_series(NET_PATH, "Strategy (net)")
    ff = (
        pd.read_csv(FF_PATH, parse_dates=["date"])
        .set_index("date")
        .sort_index()
        .asfreq("ME")
    )
    assert {"Mkt-RF", "RF"}.issubset(ff.columns), "FF file missing required columns."

    mkt = build_us_market(ff)
    rf = ff["RF"]

    idx = net.index.intersection(mkt.index).intersection(rf.index)
    net = net.loc[idx]
    mkt = mkt.loc[idx]
    rf = rf.loc[idx]

    w_net = cumulative_wealth(net, 1.0)
    w_mkt = cumulative_wealth(mkt, 1.0)

    plt.figure(figsize=(9, 5))
    plt.plot(w_net.index, w_net.values, label="Strategy (net)")
    plt.plot(w_mkt.index, w_mkt.values, label="US Market")
    plt.title("Cumulative Wealth (Monthly, start = 1.00)")
    plt.xlabel("Date")
    plt.ylabel("Wealth")
    plt.legend(loc="best")
    plt.tight_layout()
    f1 = os.path.join(FIG_DIR, "equity_curve.png")
    plt.savefig(f1, dpi=200)
    plt.close()

    dd_net = drawdown_series(net)
    plt.figure(figsize=(9, 4))
    plt.plot(dd_net.index, dd_net.values)
    plt.title("Strategy (net) Drawdown")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.tight_layout()
    f2 = os.path.join(FIG_DIR, "drawdown.png")
    plt.savefig(f2, dpi=200)
    plt.close()

    sr_net = rolling_sharpe_excess(net, rf, window=36, min_periods=12)
    sr_mkt = rolling_sharpe_excess(mkt, rf, window=36, min_periods=12)

    plt.figure(figsize=(9, 4))
    plt.plot(sr_net.index, sr_net.values, label="Strategy (net)")
    plt.plot(sr_mkt.index, sr_mkt.values, label="US Market")
    plt.axhline(0.0, linewidth=1)
    plt.title("Rolling 36-month Sharpe (excess over RF)")
    plt.xlabel("Date")
    plt.ylabel("Sharpe (36m)")
    plt.legend(loc="best")
    plt.tight_layout()
    f3 = os.path.join(FIG_DIR, "rolling_sharpe_36m.png")
    plt.savefig(f3, dpi=200)
    plt.close()

    print("[✓] Saved:")
    print(" ", f1)
    print(" ", f2)
    print(" ", f3)


if __name__ == "__main__":
    main()
