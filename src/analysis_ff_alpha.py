import os
import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
DATA = os.path.join(ROOT, "data", "cleaned")

FACTORS_CSV = os.path.join(DATA, "ff5_umd_monthly.csv")
GROSS_CSV = os.path.join(DATA, "strategy_gross_survivorship.csv")
NET_CSV = os.path.join(DATA, "strategy_net_survivorship.csv")

OUT_CSV = os.path.join(DATA, "ff_regression_table.csv")
OUT_TEX = os.path.join(DATA, "ff_regression_table.tex")

MODELS = {
    "CAPM": ["Mkt-RF"],
    "FF3": ["Mkt-RF", "SMB", "HML"],
    "FF5": ["Mkt-RF", "SMB", "HML", "RMW", "CMA"],
    "FF5+UMD": ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "UMD"],
}
HAC_LAGS = 6


def load_series(path: str, name: str) -> pd.Series:
    s = pd.read_csv(path, parse_dates=[0], index_col=0).squeeze("columns")
    if isinstance(s, pd.DataFrame):
        assert s.shape[1] == 1, f"Expected single-column series in {path}"
        s = s.iloc[:, 0]
    s.name = name
    return s.sort_index().asfreq("ME")


def load_factors(path: str) -> pd.DataFrame:
    f = (
        pd.read_csv(path, parse_dates=["date"])
        .set_index("date")
        .sort_index()
        .asfreq("ME")
    )
    need = {"RF", "Mkt-RF", "SMB", "HML", "RMW", "CMA", "UMD"}
    missing = need - set(f.columns)
    if missing:
        raise ValueError(f"Missing factor columns in {path}: {missing}")
    return f


def regress_excess(
    ret: pd.Series, fac: pd.DataFrame, cols: list[str], lags: int = HAC_LAGS
):
    y = (ret - fac["RF"]).dropna()
    X = fac.loc[y.index, cols].dropna()
    y = y.loc[X.index]
    X = sm.add_constant(X)
    res = sm.OLS(y, X, missing="drop").fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return res


def to_ann(m: float) -> float:
    return (1.0 + m) ** 12 - 1.0


def stars(p):
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.10:
        return "*"
    return ""


def main():
    fac = load_factors(FACTORS_CSV)
    gross = load_series(GROSS_CSV, "gross")
    net = load_series(NET_CSV, "net")

    idx = gross.index.intersection(net.index).intersection(fac.index)
    gross, net, fac = gross.loc[idx].dropna(), net.loc[idx].dropna(), fac.loc[idx]

    rows = []
    print("=== Factor regressions with Newey–West (HAC) SEs ===")
    for label, series in [("GROSS", gross), ("NET", net)]:
        print(f"\n--- {label} ---")
        for model_name, cols in MODELS.items():
            res = regress_excess(series, fac, cols, lags=HAC_LAGS)
            alpha_m = float(res.params["const"])
            alpha_t = float(res.tvalues["const"])
            alpha_p = float(res.pvalues["const"])
            alpha_ann = to_ann(alpha_m)
            row = {
                "Series": label,
                "Model": model_name,
                "Alpha_m": alpha_m,
                "Alpha_ann": alpha_ann,
                "Alpha_t": alpha_t,
                "Alpha_p": alpha_p,
                "R2": float(res.rsquared),
                "N": int(res.nobs),
            }
            for c in cols:
                row[f"beta_{c}"] = float(res.params.get(c, np.nan))
                row[f"t_{c}"] = float(res.tvalues.get(c, np.nan))
            rows.append(row)

            print(
                f"{model_name:7s}  α(ann)={alpha_ann: .2%}  t={alpha_t: .2f}{stars(alpha_p):>3}  "
                f"R²={res.rsquared: .3f}  n={int(res.nobs)}"
            )

    out = pd.DataFrame(rows)
    os.makedirs(DATA, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"\n[✓] Saved regression table: {OUT_CSV}")

    def fmt_pct(x):
        return "-" if pd.isna(x) else f"{x:.2%}"

    def fmt_num(x):
        return "-" if pd.isna(x) else f"{x:.2f}"

    panel = out[out["Series"] == "NET"].copy()
    panel["Alpha"] = panel.apply(
        lambda r: f"{fmt_pct(r['Alpha_ann'])} ({fmt_num(r['Alpha_t'])})", axis=1
    )
    panel = (
        panel[["Model", "Alpha", "R2", "N"]].set_index("Model").reindex(MODELS.keys())
    )

    latex = [
        r"\begin{table}[!ht]",
        r"\centering",
        r"\caption{Factor Regressions (NET returns; HAC " + str(HAC_LAGS) + r" lags)}",
        r"\label{tab:ff_regressions}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Model & $\alpha$ (annual) [t] & $R^2$ & $n$ \\",
        r"\midrule",
    ]
    for m in MODELS.keys():
        rA = panel.loc[m, "Alpha"]
        rR2 = fmt_num(panel.loc[m, "R2"])
        rN = int(panel.loc[m, "N"])
        latex.append(f"{m} & {rA} & {rR2} & {rN} \\\\")
    latex += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    with open(OUT_TEX, "w") as f:
        f.write("\n".join(latex))
    print(f"[✓] Saved LaTeX table: {OUT_TEX}")


if __name__ == "__main__":
    main()
