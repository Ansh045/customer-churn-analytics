"""Creates 3 synthetic source systems (CSV, SQLite, Excel-style CSV) with deliberate data-quality issues."""
import numpy as np, pandas as pd, sqlite3
rng = np.random.default_rng(42)
n = 5000
cust = pd.DataFrame({
 "customer_id": np.arange(1, n+1),
 "age": rng.integers(18, 70, n),
 "region": rng.choice(["North","South","East","West","Central"], n),
 "plan": rng.choice(["Basic","Standard","Premium"], n, p=[.45,.35,.2]),
 "tenure_months": rng.integers(1, 60, n)})
cust["monthly_fee"] = cust.plan.map({"Basic":300,"Standard":600,"Premium":1000})
# messy: duplicates, missing, invalid age
cust = pd.concat([cust, cust.sample(60, random_state=1)])
cust.loc[cust.sample(80, random_state=2).index, "region"] = np.nan
cust.loc[cust.sample(25, random_state=3).index, "age"] = -1
cust.to_csv("data/crm_customers.csv", index=False)

tx = pd.DataFrame({"customer_id": rng.integers(1, n+1, 40000),
 "amount": rng.gamma(2, 250, 40000).round(2),
 "discount_pct": rng.choice([0,10,20], 40000, p=[.6,.25,.15])})
tx.loc[tx.sample(150, random_state=4).index, "amount"] = np.nan
con = sqlite3.connect("data/billing.db"); tx.to_sql("transactions", con, if_exists="replace", index=False); con.close()

tk = pd.DataFrame({"customer_id": rng.integers(1, n+1, 9000),
 "resolution_days": rng.exponential(3, 9000).round(1),
 "csat": rng.integers(1, 6, 9000)})
tk.to_csv("data/support_tickets.csv", index=False)
print("data created")
