import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib
from pathlib import Path

DATA_PATH = Path("data/processed/steam_prices_full_clean.csv")

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)
df = pd.read_csv(DATA_PATH)

sales = df[df["is_on_sale"] == 1]
X = sales[["game_age_days", "month", "days_since_last_sale"]]
y = sales["discount_percent"]
model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42) #rfr
model.fit(X, y)
joblib.dump(model, MODEL_DIR / "discount_model.pkl")

print("- Discount model trained")
