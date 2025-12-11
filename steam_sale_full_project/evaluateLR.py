import pandas as pd
import numpy as np
import joblib

from sklearn.metrics import accuracy_score, brier_score_loss, mean_absolute_error, r2_score
from pathlib import Path

DATA_PATH = Path("data/processed/steam_prices_full_clean.csv")

df = pd.read_csv(DATA_PATH, parse_dates=["DateTime"])

df = df.sort_values(["game_title", "DateTime"])
X = df[["game_age_days", "month", "days_since_last_sale"]]

y = df["is_on_sale"]
cutoff = df["DateTime"].quantile(0.8)
test_idx = df["DateTime"] > cutoff

X_test = X[test_idx]
y_test = y[test_idx]
#classifier
print("\n")
print("Classifier Model:")
clf = joblib.load("models/logistic_model.pkl")

probs = clf.predict_proba(X_test)[:, 1]
preds = (probs >= 0.5).astype(int)
acc = accuracy_score(y_test, preds)
brier = brier_score_loss(y_test, probs)
print(f"Accuracy: {acc:.4f}")

print(f"Brier score: {brier:.4f}")

print("\n")
print("Regression Model:")#evaluating regression model
reg = joblib.load("models/discount_model.pkl")#adjusted for realism
sales_test = df[test_idx & (df["is_on_sale"] == 1)]

if len(sales_test) > 0:
    X_sales = sales_test[["game_age_days", "month", "days_since_last_sale"]]
    y_discount = sales_test["discount_percent"]
    discount_preds = reg.predict(X_sales)

    mae = mean_absolute_error(y_discount, discount_preds)

    r2 = r2_score(y_discount, discount_preds)
    print(f"MAE: {mae:.2f}%")
    print(f"R² Score: {r2:.4f}")
    
else:
    print("No sales in test set to evaluate.")
print("\n")