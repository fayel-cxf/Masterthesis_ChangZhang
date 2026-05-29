"""
Fit log(W_r) ~ log(DWT) regression by vessel type group
using merged List B + List C data.
"""

import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = "data"

#1. Load & merge
b = pd.read_csv(f"{DATA_DIR}/list_B_with_dwt_full.csv")
c = pd.read_csv(f"{DATA_DIR}/list_C_with_dwt.csv")

combined = pd.concat([b, c], ignore_index=True)

# Drop duplicates by IMO — keep first occurrence (list B takes priority)
n_before = len(combined)
combined = combined.drop_duplicates(subset="imo", keep="first")
n_after = len(combined)
print(f"Merged: {len(b)} (B) + {len(c)} (C) = {n_before} rows → {n_after} after dedup by IMO")
print()

#2. Quality filter
# Keep rows with valid DWT and W_r (sample_bw_mt), both > 0
mask = (combined["DWT"] > 0) & (combined["sample_bw_mt"] > 0)
dropped = (~mask).sum()
if dropped:
    print(f"Dropped {dropped} rows with DWT<=0 or W_r<=0")
df = combined[mask].copy()

print("Counts per group after merge:")
print(df["nbic_type_group"].value_counts().to_string())
print()

#3. Fit log-linear regression per group
results = []

for group, gdf in df.groupby("nbic_type_group"):
    n = len(gdf)
    log_dwt = np.log(gdf["DWT"].values)
    log_wr  = np.log(gdf["sample_bw_mt"].values)

    if n < 5:
        print(f"[SKIP] {group}: only {n} samples — too few to fit")
        continue

    slope, intercept, r, p, se = stats.linregress(log_dwt, log_wr)
    r2 = r ** 2

    # 95% CI on slope
    t_crit = stats.t.ppf(0.975, df=n - 2)
    slope_ci_lo = slope - t_crit * se
    slope_ci_hi = slope + t_crit * se

    # Residual std (in log space) — used for uncertainty propagation
    log_wr_pred = intercept + slope * log_dwt
    residuals = log_wr - log_wr_pred
    rmse_log = np.sqrt(np.mean(residuals ** 2))

    results.append({
        "group":        group,
        "n":            n,
        "intercept":    round(intercept, 4),
        "slope":        round(slope, 4),
        "slope_CI_lo":  round(slope_ci_lo, 4),
        "slope_CI_hi":  round(slope_ci_hi, 4),
        "R2":           round(r2, 4),
        "p_value":      round(p, 4),
        "RMSE_log":     round(rmse_log, 4),
    })

results_df = pd.DataFrame(results).sort_values("group")

#4. Print summary
print("=" * 80)
print("log(W_r) = intercept + slope × log(DWT)   [W_r in mt, DWT in t]")
print("=" * 80)
for _, row in results_df.iterrows():
    print(f"\n{row['group']}  (n={row['n']})")
    print(f"  log(W_r) = {row['intercept']} + {row['slope']} × log(DWT)")
    print(f"  Slope 95% CI: [{row['slope_CI_lo']}, {row['slope_CI_hi']}]")
    print(f"  R² = {row['R2']},  p = {row['p_value']},  RMSE(log) = {row['RMSE_log']}")

#5. Save results
out_path = f"{DATA_DIR}/log_regression_results.csv"
results_df.to_csv(out_path, index=False)
print(f"\nResults saved to {out_path}")

# Also save the merged dataset used for fitting
merged_path = f"{DATA_DIR}/list_BC_merged.csv"
df.to_csv(merged_path, index=False)
print(f"Merged dataset saved to {merged_path}")
