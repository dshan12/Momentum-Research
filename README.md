# Momentum-Research : Re-examining 12-Month Equity Momentum (2005 – 2024)

[![DOI](https://zenodo.org/badge/1018743111.svg)](https://doi.org/10.5281/zenodo.16702629)
![Python](https://img.shields.io/badge/python-3.11-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

A fully reproducible study that revisits the classic **12 – 1 momentum strategy** in the S&P 500 from January 2005 through June 2024.  
It answers three questions:

1. **Does large-cap momentum still work after survivorship, look-ahead, and realistic transaction costs?**  
2. **How stable is the edge across look-back horizons and portfolio sizes?**  
3. **What risk factors and market regimes drive performance (or the lack thereof)?**

All code is released under the MIT license; nothing herein constitutes investment advice.

▶ **No API keys required.** Data download uses *yfinance*; a 500-row sample CSV ships inside `data/` for offline tests.

---

## Key Features

| Feature | What it does | Why it matters |
|---------|--------------|----------------|
| **Bias controls** | Rebuilds historical S&P 500 membership; lags signals one trading day | Removes survivorship & look-ahead |
| **Transaction-cost model** | User-settable round-trip bps; default = 10 bps | Tests real-world implementability |
| **Robustness sweep** | Grid search over look-back horizons & top-N winners/losers | Detects parameter cherry-picking |
| **Risk diagnostics** | CAPM regression, bootstrap Sharpe, rolling 36-mo metrics | Quantifies statistical significance |

---

## Results Snapshot

<p align="center">
  <img src="figures/equity_curve_net_costs.png" width="600" alt="Equity curve">
</p>

*Figure 1 – Momentum strategy vs. SPY net of 10 bps costs.*

---

## How to Cite

> **Sathish Kumar, D.** (2025). _Momentum-Research: Re-examining 12-Month Equity Momentum (Version 1.0.0)_. Zenodo. https://doi.org/10.5281/zenodo.16702629

A separate DOI for the working-paper PDF is listed under **Related Identifiers** in the Zenodo record.

---

## License

This project is licensed under the **MIT License** – see [`LICENSE`](LICENSE) for details.
