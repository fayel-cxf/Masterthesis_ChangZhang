"""
section_4_3_analysis.py
========================

Inputs (all in data/):
  combined_synthetic_corpus.csv 
  risk_observed.csv 

Outputs: printed tables + optional CSV exports (see EXPORT_CSV flag).
"""

import os
import numpy as np
import pandas as pd
from scipy import stats

#configuration
_BASE        = os.path.dirname(os.path.abspath(__file__))
_OUT         = os.path.join(_BASE, "..", "output")
CORPUS_PATH  = os.path.join(_OUT, "combined_synthetic_corpus.csv")
OBS_PATH     = os.path.join(_OUT, "risk_observed.csv")
EXPORT_CSV   = False   # set True to write table_4_3.csv … table_4_6.csv

PORT_ORDER = [
    "AUCKLAND", "TAURANGA", "GISBORNE", "WELLINGTON", "NAPIER",
    "NELSON", "PORT CHALMERS", "NEW PLYMOUTH", "LYTTELTON", "TIMARU",
]

def fmt(x, scale=1e4, dp=3):
    return f"{x * scale:.{dp}f}"


def ks2(a, b):
    return stats.ks_2samp(a, b).statistic



corpus = pd.read_csv(CORPUS_PATH)
obs    = pd.read_csv(OBS_PATH)

t1 = corpus[corpus["track"] == "Track1"].copy()
t2 = corpus[corpus["track"] == "Track2"].copy()

print("=" * 70)
print("§4.3.1  COMBINED CORPUS CONSTRUCTION")
print("=" * 70)

n_t1, n_t2, n_total = len(t1), len(t2), len(corpus)
m_t1  = t1["p_invasion"].mean()
m_t2  = t2["p_invasion"].mean()
m_obs = obs["p_invasion"].mean()
m_pool = corpus["p_invasion"].mean()

ks_t1_obs  = ks2(t1["p_invasion"].values, obs["p_invasion"].values)
ks_t2_obs  = ks2(t2["p_invasion"].values, obs["p_invasion"].values)
ks_t1_t2   = ks2(t1["p_invasion"].values, t2["p_invasion"].values)
ks_t1t2_p  = stats.ks_2samp(t1["p_invasion"].values, t2["p_invasion"].values).pvalue

print(f"Track 1  : n={n_t1:,}  mean={fmt(m_t1)}e-4  Fréchet(vs obs)={ks_t1_obs:.3f}")
print(f"Track 2  : n={n_t2:,}  mean={fmt(m_t2)}e-4  Fréchet(vs obs)={ks_t2_obs:.3f}")
print(f"Observed : n={len(obs):,}  mean={fmt(m_obs)}e-4")
print(f"Pooled   : n={n_total:,}  mean={fmt(m_pool)}e-4  ({(m_pool/m_obs-1)*100:+.1f}% vs obs)")
print(f"KS(T1 vs T2) = {ks_t1_t2:.3f}  p={ks_t1t2_p:.2e}")
inter_gap = abs(m_t1 - m_t2) / m_obs * 100
t1_gap    = (m_t1 / m_obs - 1) * 100
t2_gap    = (m_t2 / m_obs - 1) * 100
print(f"T1 vs obs: {t1_gap:+.1f}%   T2 vs obs: {t2_gap:+.1f}%   inter-track gap: {inter_gap:.1f}%")


#§4.3.2  PORT-LEVEL RANKING
print("\n" + "=" * 70)
print("§4.3.2  TABLE 4.3 — Port-level invasion risk ranking")
print("=" * 70)

port_syn = (corpus.groupby("nz_dest_port")["p_invasion"]
            .agg(mean_syn="mean", n_syn="count").reset_index())
port_obs = (obs.groupby("nz_dest_port")["p_invasion"]
            .agg(mean_obs="mean", n_obs="count").reset_index())

tbl43 = port_syn.merge(port_obs, on="nz_dest_port", how="left")
tbl43["delta_pct"] = (tbl43["mean_syn"] / tbl43["mean_obs"] - 1) * 100
tbl43 = tbl43.sort_values("mean_syn", ascending=False).reset_index(drop=True)
tbl43.index += 1

print(f"{'Rank':<5} {'Port':<16} {'Comb mean(×1e-4)':<20} {'Obs mean(×1e-4)':<18} {'Δ(%)':<10} {'n_comb':<10} {'n_obs'}")
print("-" * 85)
for rank, row in tbl43.iterrows():
    print(f"{rank:<5} {row['nz_dest_port']:<16} {row['mean_syn']*1e4:<20.3f} "
          f"{row['mean_obs']*1e4:<18.3f} {row['delta_pct']:<10.1f} "
          f"{int(row['n_syn']):<10} {int(row['n_obs'])}")

if EXPORT_CSV:
    tbl43.to_csv(os.path.join(_OUT, "table_4_3.csv"), index_label="rank")


#§4.3.3T1 vs T2 AGREEMENT PER PORT
print("\n" + "=" * 70)
print("§4.3.3  TABLE 4.4 — Track 1 vs Track 2 agreement per port")
print("=" * 70)

rows44 = []
for port in tbl43["nz_dest_port"]:
    g1 = t1[t1["nz_dest_port"] == port]["p_invasion"].values
    g2 = t2[t2["nz_dest_port"] == port]["p_invasion"].values
    obs_mean = port_obs.loc[port_obs["nz_dest_port"] == port, "mean_obs"].values
    obs_m = obs_mean[0] if len(obs_mean) else np.nan
    if len(g1) < 5 or len(g2) < 5:
        ks_val = np.nan
    else:
        ks_val = ks2(g1, g2)
    gap_rel = abs(g1.mean() - g2.mean()) / obs_m * 100 if not np.isnan(obs_m) else np.nan
    if ks_val < 0.10:
        conf = "High"
    elif ks_val < 0.22:
        conf = "Moderate"
    else:
        conf = "Low"
    rows44.append({
        "port": port,
        "t1_mean": g1.mean(),
        "t2_mean": g2.mean(),
        "gap_pct_obs": gap_rel,
        "ks": ks_val,
        "confidence": conf,
    })

tbl44 = pd.DataFrame(rows44).sort_values("ks")
print(f"{'Port':<16} {'T1 mean(×1e-4)':<16} {'T2 mean(×1e-4)':<16} {'|Δ|/obs(%)':<12} {'KS':<8} Confidence")
print("-" * 78)
for _, row in tbl44.iterrows():
    print(f"{row['port']:<16} {row['t1_mean']*1e4:<16.3f} {row['t2_mean']*1e4:<16.3f} "
          f"{row['gap_pct_obs']:<12.1f} {row['ks']:<8.3f} {row['confidence']}")

if EXPORT_CSV:
    tbl44.to_csv(os.path.join(_OUT, "table_4_4.csv"), index=False)


#VESSEL-TYPE DECOMPOSITION
print("\n" + "=" * 70)
print("§4.3.4  TABLE 4.5 — Vessel-type risk decomposition (top ports)")
print("=" * 70)

FOCUS_PORTS = ["AUCKLAND", "TAURANGA", "GISBORNE", "WELLINGTON", "NEW PLYMOUTH"]

rows45 = []
for port in FOCUS_PORTS:
    sub = corpus[corpus["nz_dest_port"] == port].copy()
    port_total_risk = sub["p_invasion"].sum()
    for vt, grp in sub.groupby("vessel_type"):
        voy_share = len(grp) / len(sub) * 100
        risk_share = grp["p_invasion"].sum() / port_total_risk * 100
        rows45.append({
            "port": port,
            "vessel_type": vt,
            "voyage_share": voy_share,
            "mean_pinv": grp["p_invasion"].mean(),
            "risk_share": risk_share,
            "uplift": risk_share - voy_share,
        })

tbl45 = pd.DataFrame(rows45).sort_values(["port", "risk_share"], ascending=[True, False])

print(f"{'Port':<16} {'VesselType':<16} {'VoyShare%':<12} {'MeanP(×1e-4)':<16} {'RiskShare%':<12} {'Uplift(pp)'}")
print("-" * 80)
prev_port = None
for _, row in tbl45.iterrows():
    if row["port"] != prev_port:
        if prev_port is not None:
            print()
        prev_port = row["port"]
    flag = " ↑" if row["uplift"] >= 0.5 else "  "
    print(f"{row['port']:<16} {row['vessel_type']:<16} {row['voyage_share']:<12.1f} "
          f"{row['mean_pinv']*1e4:<16.3f} {row['risk_share']:<12.1f} {row['uplift']:+.1f}{flag}")

if EXPORT_CSV:
    tbl45.to_csv(os.path.join(_OUT, "table_4_5.csv"), index=False)


#§4.3.5  WEIGHTING-SCENARIO ROBUSTNESS
print("\n" + "=" * 70)
print("§4.3.5  TABLE 4.6 — Port ranking under alternative Track weighting scenarios")
print("=" * 70)

SCENARIOS = {
    "S0_equal":   (1.0, 1.0),
    "S1_T1heavy": (2.0, 1.0),
    "S2_T2heavy": (1.0, 2.0),
    "S3_T1only":  (1.0, 0.0),
    "S4_T2only":  (0.0, 1.0),
}

t1_port = t1.groupby("nz_dest_port")["p_invasion"].mean().rename("t1_mean")
t2_port = t2.groupby("nz_dest_port")["p_invasion"].mean().rename("t2_mean")
t1_n    = t1.groupby("nz_dest_port").size().rename("n_t1")
t2_n    = t2.groupby("nz_dest_port").size().rename("n_t2")
port_df = pd.concat([t1_port, t2_port, t1_n, t2_n], axis=1).fillna(0)

scenario_means = {}
for sname, (w1, w2) in SCENARIOS.items():
    wt_sum = w1 * port_df["n_t1"] + w2 * port_df["n_t2"]
    wt_mean = (w1 * port_df["n_t1"] * port_df["t1_mean"] +
               w2 * port_df["n_t2"] * port_df["t2_mean"]) / wt_sum
    scenario_means[sname] = wt_mean

means_df = pd.DataFrame(scenario_means)
ranks_df = means_df.rank(ascending=False).astype(int)
ranks_df["range"] = ranks_df.max(axis=1) - ranks_df.min(axis=1)
ranks_df = ranks_df.loc[ranks_df["S0_equal"].sort_values().index]

print(f"{'Port':<16} {'S0':>4} {'S1':>4} {'S2':>4} {'S3':>4} {'S4':>4} {'Range':>6}")
print("-" * 50)
for port, row in ranks_df.iterrows():
    print(f"{port:<16} {row['S0_equal']:>4} {row['S1_T1heavy']:>4} "
          f"{row['S2_T2heavy']:>4} {row['S3_T1only']:>4} {row['S4_T2only']:>4} "
          f"{row['range']:>6}")

# Spearman correlations vs S0
print("\nSpearman rank correlation vs S0 (equal-weight baseline):")
s0_ranks = ranks_df["S0_equal"].values
for sname in ["S1_T1heavy", "S2_T2heavy", "S3_T1only", "S4_T2only"]:
    rho, pv = stats.spearmanr(s0_ranks, ranks_df[sname].values)
    print(f"  {sname}: ρ = {rho:.3f}  p = {pv:.3e}")

if EXPORT_CSV:
    ranks_df.to_csv(os.path.join(_OUT, "table_4_6.csv"))

print("\n[Done]")
