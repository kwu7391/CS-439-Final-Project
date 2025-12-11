import pandas as pd

DATA_PATH = "data/processed/steam_prices_full_clean.csv"
df = pd.read_csv(DATA_PATH)

print("\nSale Rates")
print(df.groupby("game_title")["is_on_sale"].mean().sort_values(ascending=False))
print("\nAvg sale Discount")

print(df[df["is_on_sale"] == 1].groupby("game_title")["discount_percent"].mean().sort_values(ascending=False))
print("\nFrequency of sales Every Month")
print(df.groupby("month")["is_on_sale"].mean())

print("\nDistribution")
print(df["discount_percent"].describe())
print("\n - EDA done!")
