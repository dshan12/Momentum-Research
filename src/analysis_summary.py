import os
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
DATA = os.path.join(ROOT, "data", "cleaned")
PAPER_TABLES = os.path.join(ROOT, "paper", "tables")

NET_PATH = os.path.join(DATA, "strategy_net_survivorship.csv")
FF_PATH = os.path.join(DATA, "ff5_umd_monthly.csv")

OUT_CSV = os.path.join(DATA, "summary_stats.csv")
OUT_TEX = os.path.join(PAPER_TABLES, "summary_stats.tex")


def load_series(path: str, name: str) -> pd.Series:
    s = pd.read_csv(path, parse_dates=[0], index_col=0).squeeze("columns")
    if isinstance(s, pd.DataFrame):
        assert s.shape[1] == 1, f"Expected 1 column in {path}"
        s = s.iloc[:, 0]
    s.name = name
    return s.sort_index().asfreq("ME")


def max_drawdown(r: pd.Series) -> float:
    w = (1.0 + r.fillna(0.0)).cumprod()
    peak = w.cummax()
    dd = (w / peak) - 1.0
    return float(dd.min())


def sharpe_ratio(r: pd.Series, rf_annual: float = 0.02) -> float:
    rf_m = (1.0 + rf_annual) ** (1.0 / 12.0) - 1.0
    ex = r - rf_m
    return (ex.mean() / (ex.std() + 1e-12)) * np.sqrt(12.0)


def ann_return(r: pd.Series) -> float:
    return (1.0 + r.mean()) ** 12 - 1.0


def build_benchmark_from_factors(ff: pd.DataFrame) -> pd.Series:
    # Market total return proxy = Mkt-RF + RF (decimal monthly)
    mkt = (ff["Mkt-RF"] + ff["RF"]).rename("us_market")
    return mkt.asfreq("ME")


def summary_table(series: dict[str, pd.Series]) -> pd.DataFrame:
    rows = []
    for label, r in series.items():
        r = r.dropna()
        rows.append(
            {
                "Series": label,
                "N": int(r.shape[0]),
                "Mean (m)": r.mean(),
                "Vol (m)": r.std(),
                "Sharpe (2% rf)": sharpe_ratio(r),
                "Ann Return": ann_return(r),
                "Ann Vol": r.std() * np.sqrt(12.0),
                "Skew": r.skew(),
                "Kurtosis": r.kurtosis(),  # excess kurtosis (Fisher)
                "Max DD": max_drawdown(r),
            }
        )
    df = pd.DataFrame(rows).set_index("Series")
    return df


def save_latex(table: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fmt = {
        "N": "{:d}".format,
        "Mean (m)": "{:.3%}".format,
        "Vol (m)": "{:.3%}".format,
        "Sharpe (2% rf)": "{:.2f}".format,
        "Ann Return": "{:.2%}".format,
        "Ann Vol": "{:.2%}".format,
        "Skew": "{:.2f}".format,
        "Kurtosis": "{:.2f}".format,
        "Max DD": "{:.2%}".format,
    }
    # apply formatting for LaTeX-friendly text
    disp = table.copy()
    for c, f in fmt.items():
        disp[c] = [f(x) for x in disp[c]]

    lines = []
    lines += [
        r"\begin{table}[!ht]",
        r"\centering",
        r"\caption{Summary statistics (monthly, decimal returns)}",
        r"\label{tab:summary_stats}",
        r"\begin{tabular}{lrrrrrrrrr}",
        r"\toprule",
        r"Series & $N$ & Mean (m) & Vol (m) & Sharpe & Ann Ret & Ann Vol & Skew & Kurtosis & Max DD \\",
        r"\midrule",
    ]
    for idx, row in disp.iterrows():
        lines.append(
            f"{idx} & {row['N']} & {row['Mean (m)']} & {row['Vol (m)']} & {row['Sharpe (2% rf)']} & "
            f"{row['Ann Return']} & {row['Ann Vol']} & {row['Skew']} & {row['Kurtosis']} & {row['Max DD']} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    with open(path, "w") as f:
        f.write("\n".join(lines))


def main():
    # load net strategy returns
    net = load_series(NET_PATH, "Strategy (net)")
    # load factors & build US market proxy
    ff = (
        pd.read_csv(FF_PATH, parse_dates=["date"])
        .set_index("date")
        .sort_index()
        .asfreq("ME")
    )
    bench = build_benchmark_from_factors(ff).rename("US Market")

    # align on overlap
    idx = net.index.intersection(bench.index)
    net, bench = net.loc[idx], bench.loc[idx]

    # build table
    table = summary_table({"Strategy (net)": net, "US Market": bench})

    # save
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    table.to_csv(OUT_CSV)
    save_latex(table, OUT_TEX)

    # console print (ASCII)
    disp = table.copy()
    print("\n=== Summary statistics (monthly series) ===")
    with pd.option_context("display.float_format", "{:.6f}".format):
        print(disp)

    print(f"\n[✓] Saved CSV: {OUT_CSV}")
    print(f"[✓] Saved LaTeX: {OUT_TEX}")


if __name__ == "__main__":
    main()
