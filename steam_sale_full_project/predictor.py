import pandas as pd
import joblib

clf = joblib.load("models/logistic_model.pkl")

reg = joblib.load("models/discount_model.pkl")
df = pd.read_csv("data/processed/steam_prices_full_clean.csv", parse_dates=["DateTime"])
#load
games = sorted(df["game_title"].unique())#display games
print("\nAvailable Games:")
for i, g in enumerate(games, 1):
    print(f"{i}. {g}")
GAME = input("\n Please type game title: ").strip()#input
if GAME not in games:
    raise ValueError("Invalid.")

game_df = df[df["game_title"] == GAME].sort_values("DateTime")
latest = game_df.dropna(subset=["price"]).iloc[-1]

X = pd.DataFrame([{"game_age_days": latest["game_age_days"],#features
    "month": latest["month"], "days_since_last_sale": latest["days_since_last_sale"]}])
prob = clf.predict_proba(X)[0][1]#predictions

expected_discount = reg.predict(X)[0]

if latest["is_on_sale"] == 1:
    current_discount = latest["discount_percent"]
    if current_discount >= expected_discount * 0.9: #check if currently osale
        decision = "Buy game"
    else:
        decision = "Wait for sale"
else:
    if prob >= 0.5:#not on sale
        decision = "Wait for sale"
    else:
        decision = "Buy game"
current_price = latest["price"]
if latest["is_on_sale"] == 1:
    full_price = latest["original_price"]
else:
    full_price = current_price
expected_sale_price = full_price * (1 - expected_discount/100)

potential_savings = current_price - expected_sale_price
print(f"\nGame: {GAME}")
print(f"Price: ${current_price:.2f}")
if latest["is_on_sale"] == 1:
    print("Currently on sale!")

    print(f"Discount: {latest['discount_percent']:.1f}% off")
else:
    print("Not currently on sale.")
print(f"Days Since Sale: {int(latest['days_since_last_sale'])}")

print(f"What I recommend: {decision}")
print(f"\n")
if latest["is_on_sale"] != 1:
    print(f"Sale Chance: {prob:.1%}")
    print(f"Estimated Discount for future sale: {expected_discount:.1f}%")


