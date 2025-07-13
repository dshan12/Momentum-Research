import pandas as pd

prices = pd.read_csv(
    "./data/cleaned/cleaned_monthly_prices.csv", index_col=0, parse_dates=True
)

prices.head()

print("Missing values per ticke:", prices.isna().sum().head())
print("Shape:", prices.shape)
