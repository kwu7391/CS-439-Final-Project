import pandas as pd
import matplotlib.pyplot as plt
import os

DATA_PATH = "data/processed/steam_prices_full_clean.csv"

OUTPUT_DIR = "figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)
monthly = df.groupby("month")["is_on_sale"].mean()
plt.figure()
monthly.plot(kind="bar")
plt.title("Monthly Prob. of Sale")#first graph
plt.xlabel("Month")

plt.ylabel("Sale Prob.")
plt.tight_layout()

plt.savefig(f"{OUTPUT_DIR}/monthly_prob.png")
plt.close()
sales = df[df["is_on_sale"] == 1]
plt.figure()
plt.hist(sales["discount_percent"], bins=25)
plt.title("Discount Distribution")#second graph (distribution)
plt.xlabel("% Discount")
plt.ylabel("Freq")

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/distribution.png")
plt.close()


print("\n - Visualizations created! (they're in /figures/) ")
