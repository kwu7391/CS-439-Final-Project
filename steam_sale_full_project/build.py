import pandas as pd
import numpy as np
from pathlib import Path

RAW_DATA_DIR = Path("data/raw_games")

OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "steam_prices_full_clean.csv"

all_games = []
for file in RAW_DATA_DIR.glob("*.csv"):
    game_title = file.stem
    df = pd.read_csv(file)

    df.columns = [c.strip() for c in df.columns]
    if "price" in df.columns:#for getting price col.
        price_col = "price"
    elif "Final price" in df.columns:
        price_col = "Final price"
    elif "final_price" in df.columns:
        price_col = "final_price"
    else:
        raise ValueError(f"Price column not found...")
    df["DateTime"] = pd.to_datetime(df["DateTime"])
    df = df.sort_values("DateTime").reset_index(drop=True)

    df["price"] = df[price_col].astype(float)
    df["release_date"] = df["DateTime"].min()

    df["game_age_days"] = (df["DateTime"] - df["release_date"]).dt.days
    df["month"] = df["DateTime"].dt.month
    df["original_price"] = (df["price"].rolling(60, min_periods=1).apply(lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[-1]))
    df["is_on_sale"] = (df["price"] < df["original_price"]).astype(int)

    df["discount_percent"] = np.where(df["is_on_sale"] == 1,(1 - df["price"] / df["original_price"]) * 100,0.0,)
    df["last_sale_date"] = df["DateTime"].where(df["is_on_sale"] == 1)
    df["last_sale_date"] = df["last_sale_date"].ffill()

    df["days_since_last_sale"] = (df["DateTime"] - df["last_sale_date"]).dt.days
    df["days_since_last_sale"] = df["days_since_last_sale"].fillna(999)
    df["game_title"] = game_title
    all_games.append(df)

master_df = pd.concat(all_games, ignore_index=True)

master_df.to_csv(OUTPUT_FILE, index=False)
print("\n- Dataset built")
