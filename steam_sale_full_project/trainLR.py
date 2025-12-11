import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
import joblib
from pathlib import Path

DATA_PATH = Path("data/processed/steam_prices_full_clean.csv")
MODEL_DIR = Path("models")

MODEL_DIR.mkdir(exist_ok=True)
df = pd.read_csv(DATA_PATH, parse_dates=["DateTime"])#load the data
df = df.sort_values(["game_title", "DateTime"])
#features/labels
X = df[["game_age_days", "month", "days_since_last_sale"]]

y = df["is_on_sale"]
cutoff = df["DateTime"].quantile(0.8)#time split
test_idx = df["DateTime"] > cutoff
train_idx = df["DateTime"] <= cutoff

X_train, y_train = X[train_idx], y[train_idx]

X_test, y_test = X[test_idx], y[test_idx]
#models and calibration
base = LogisticRegression(max_iter=2000,C=0.3, solver="lbfgs")

model = CalibratedClassifierCV(base,method="isotonic",cv=5)
model.fit(X_train, y_train)
joblib.dump(model, MODEL_DIR / "logistic_model.pkl")

print("- Classifier model trained")
