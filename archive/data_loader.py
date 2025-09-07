import pandas as pd
import yfinance as yf

# Constants
WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
START_DATE = "2005-01-01"
END_DATE = "2024-12-31"


def fetch_sp500_tickers():
    """
    Scrapes the Wikipedia page for the S&P 500 and returns a list of tickers
    """
    table = pd.read_html(WIKI_URL, header=0)[0]
    # Convert all . to -
    tickers = table.Symbol.str.replace(".", "-", regex=False).tolist()
    return tickers


def download_monthly_prices(tickers):
    """
    Downloads the monthly prices for the given tickers and returns a dataframe
    """
    prices = yf.download(
        tickers,
        start=START_DATE,
        end=END_DATE,
        interval="1mo",
        progress=False,
        auto_adjust=True,
    )["Close"]
    return prices


def clean_prices(prices, min_data_pct=0.9):
    """
    Drop tickers with > (1-min_data_pct) missing data,
    then forward-fill missing data
    """
    min_obs = int(min_data_pct * len(prices))
    prices = prices.dropna(thresh=min_obs, axis=1)
    prices = prices.ffill()
    return prices


if __name__ == "__main__":
    # 1. Fetch tickers
    print("Fetching S&P 500 tickers...")
    tickers = fetch_sp500_tickers()
    print(f"  -> {len(tickers)} tickers found.")

    # 2. Download prices
    print("Downloading prices...")
    prices = download_monthly_prices(tickers)
    print("  -> Raw data shape:", prices.shape)

    # 3. Clean data
    print("Cleaning data...")
    clean = clean_prices(prices)
    print("  -> Clean data shape:", clean.shape)

    # 4. Save data
    clean.to_csv("data/cleaned/cleaned_monthly_prices.csv")
    print("Data saved to data/cleaned/cleaned_monthly_prices.csv")
