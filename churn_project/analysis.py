import pandas as pd, numpy as np, sqlite3, json
from scipy import stats
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report
rng = np.random.default_rng(7)

# 1. LOAD from 3 sources
crm = pd.read_csv("data/crm_customers.csv")
tk = pd.read_csv("data/support_tickets.csv")
con = sqlite3.connect("data/billing.db")
# SQL aggregation in the billing system
bill = pd.read_sql("""SELECT customer_id, COUNT(*) AS n_tx, ROUND(AVG(amount),2) AS avg_amt,
  ROUND(AVG(discount_pct),2) AS avg_disc FROM transactions WHERE amount IS NOT NULL
  GROUP BY customer_id""", con)
con.close()

# 2. CLEAN
log = {"crm_rows_raw": len(crm)}
crm = crm.drop_duplicates("customer_id"); log["duplicates_removed"] = log["crm_rows_raw"] - len(crm)
log["invalid_age_fixed"] = int((crm.age < 0).sum()); crm.loc[crm.age < 0, "age"] = crm.age.median()
log["missing_region_filled"] = int(crm.region.isna().sum()); crm["region"] = crm.region.fillna("Unknown")
tkg = tk.groupby("customer_id").agg(n_tickets=("csat","size"), avg_res_days=("resolution_days","mean"), avg_csat=("csat","mean")).reset_index()

# 3. INTEGRATE (left join keeps all customers)
df = crm.merge(bill, on="customer_id", how="left").merge(tkg, on="customer_id", how="left")
for c in ["n_tx","n_tickets"]: df[c] = df[c].fillna(0)
for c in ["avg_amt","avg_disc","avg_res_days","avg_csat"]: df[c] = df[c].fillna(df[c].median())

# 4. Target: simulated churn driven by realistic business factors
z = (-1.2 + .35*df.n_tickets + .25*df.avg_res_days - .5*df.avg_csat + .02*df.avg_disc
     - .02*df.tenure_months + .0*df.age + rng.normal(0, .8, len(df)) + 1.2)
df["churned"] = (rng.random(len(df)) < 1/(1+np.exp(-z))).astype(int)

# 5. VALIDATION CHECKS
checks = {"no_duplicate_ids": df.customer_id.is_unique, "age_in_range": df.age.between(18,70).all(),
 "no_nulls_in_features": not df[["avg_amt","avg_csat","region"]].isna().any().any(),
 "row_count_matches_crm": len(df)==crm.shape[0], "churn_is_binary": set(df.churned.unique())<={0,1},
 "fee_positive": (df.monthly_fee>0).all(), "tenure_positive": (df.tenure_months>0).all(),
 "join_no_row_explosion": len(df)==crm.customer_id.nunique()}
log["validation_checks"] = f"{sum(checks.values())}/{len(checks)} passed"; assert all(checks.values())

# 6. STATISTICAL TESTS
ch, st = df[df.churned==1], df[df.churned==0]
t_res = stats.ttest_ind(ch.avg_res_days, st.avg_res_days, equal_var=False)
anova = stats.f_oneway(*[g.churned for _, g in df.groupby("plan")])
chi = stats.chi2_contingency(pd.crosstab(df.region, df.churned))
corr = df[["churned","n_tickets","avg_res_days","avg_csat","tenure_months","avg_disc"]].corr()["churned"].round(3).to_dict()
rate = df.churned.mean()
log.update({"churn_rate": round(rate,3),
 "ttest_resolution_days": {"churned_mean": round(ch.avg_res_days.mean(),2), "retained_mean": round(st.avg_res_days.mean(),2), "p": float(f"{t_res.pvalue:.2e}")},
 "anova_churn_by_plan_p": float(f"{anova.pvalue:.3f}"), "chi2_region_p": round(chi[1],3), "correlation_with_churn": corr})

# 7. MODELS (train/test + 5-fold CV)
feats = ["age","tenure_months","monthly_fee","n_tx","avg_amt","avg_disc","n_tickets","avg_res_days","avg_csat"]
X, y = df[feats], df.churned
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.25, stratify=y, random_state=42)
lr = LogisticRegression(max_iter=2000).fit((Xtr-Xtr.mean())/Xtr.std(), ytr)
rf = RandomForestClassifier(300, max_depth=6, random_state=42).fit(Xtr, ytr)
auc_lr = roc_auc_score(yte, lr.predict_proba((Xte-Xtr.mean())/Xtr.std())[:,1])
auc_rf = roc_auc_score(yte, rf.predict_proba(Xte)[:,1])
cv = cross_val_score(RandomForestClassifier(300, max_depth=6, random_state=42), X, y, cv=5, scoring="roc_auc")
imp = pd.Series(rf.feature_importances_, feats).sort_values(ascending=False).round(3)
log.update({"auc_logistic": round(auc_lr,3), "auc_random_forest": round(auc_rf,3),
 "rf_cv_auc_mean": round(cv.mean(),3), "rf_cv_auc_std": round(cv.std(),3), "top_features": imp.head(4).to_dict()})

# 8. KPI MONITORING + BI EXPORT
df["risk_score"] = rf.predict_proba(X)[:,1]
df["risk_band"] = pd.cut(df.risk_score, [0,.3,.6,1], labels=["Low","Medium","High"])
kpi = df.groupby("plan").agg(customers=("customer_id","size"), churn_rate=("churned","mean"), avg_csat=("avg_csat","mean"), avg_res_days=("avg_res_days","mean")).round(3)
kpi.to_csv("outputs/kpi_by_plan.csv"); df.groupby("region").churned.mean().round(3).to_csv("outputs/churn_by_region.csv")
df.to_csv("outputs/powerbi_customer_master.csv", index=False)
hi = df[df.risk_band=="High"]; log["high_risk_customers"] = len(hi)
log["high_risk_actual_churn"] = round(hi.churned.mean(),3)
log["monthly_revenue_at_risk_high_band"] = int(hi.monthly_fee.sum())
# fast resolution what-if
fast = df[df.avg_res_days<=df.avg_res_days.median()].churned.mean(); slow = df[df.avg_res_days>df.avg_res_days.median()].churned.mean()
log["churn_fast_vs_slow_resolution"] = [round(fast,3), round(slow,3)]
json.dump(log, open("outputs/results_summary.json","w"), indent=2, default=str)
print(json.dumps(log, indent=2, default=str))
