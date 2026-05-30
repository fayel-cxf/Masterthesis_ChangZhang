"""
layer3_post_assessment.py
=========================
Reproduces every quantitative result in report.md (Layer 3 Post-Assessment).

Sections covered
  §1   Risk distribution overview
  §2   Gap(i) — tail risk coverage (TMR, P95 ratio, max extension)
  §3   Layer 3 validation (Fréchet distance, KS by vessel type, KS by n_ports)
  §4   Gap(ii) — multi-stop monotonicity (per-hop stats, component breakdown,
       per-hop KS, Spearman, Mann-Whitney)
  §5   NZ destination distribution & CVAE QC (JSD, TVD, risk by port)
  §6   Risk breakdown by voyage length
  §7   Risk breakdown by vessel type
  §8   Pooled corpus summary
  App A  Decay ratio
  App B  Gap(ii) by voyage length (synthetic)

Inputs  (output/):
  risk_observed.csv
  track2_risk_synthetic.csv

Usage:
  python layer3_post_assessment.py
"""

import ast
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import jensenshannon

warnings.filterwarnings("ignore")

import os
BASE   = os.path.dirname(os.path.abspath(__file__))
OUTPUT = os.path.join(BASE, "..", "output")

# ── Load data ─────────────────────────────────────────────────────────────────
obs = pd.read_csv(os.path.join(OUTPUT, "risk_observed.csv"))
syn = pd.read_csv(os.path.join(OUTPUT, "track2_risk_synthetic.csv"))

N_OBS, N_SYN = len(obs), len(syn)

PORTS = [
    "AUCKLAND", "TAURANGA", "NEW PLYMOUTH", "LYTTELTON", "NELSON",
    "NAPIER", "GISBORNE", "WELLINGTON", "TIMARU", "PORT CHALMERS",
]
HOP_RANGE = [1, 2, 3, 4, 5, 6]
FRECHET_THRESHOLD = 0.15

def parse(s):
    try:
        return ast.literal_eval(str(s))
    except Exception:
        return []

def sep(title=""):
    w = 72
    if title:
        print(f"\n{'═'*4}  {title}  {'═'*(w - len(title) - 6)}")
    else:
        print("─" * w)

# ── Build per-hop DataFrames ──────────────────────────────────────────────────
def build_hop_df(df):
    rows = []
    for _, r in df.iterrows():
        pa = parse(r["p_alien_per_port"])
        pi = parse(r["p_intro_per_port"])
        pe = parse(r["p_estab_per_port"])
        pr = parse(r["pr_inv_per_port"])
        n  = len(pr)
        for k, (a, i, e, p) in enumerate(zip(pa, pi, pe, pr)):
            rows.append({
                "n_ports":    int(r["n_ports"]),
                "hops_to_nz": n - k,
                "p_alien": float(a), "p_intro": float(i),
                "p_estab": float(e), "pr_inv":  float(p),
            })
    return pd.DataFrame(rows)

print("Building hop-level data …", end=" ", flush=True)
obs_hop = build_hop_df(obs)
syn_hop = build_hop_df(syn)
print("done")

# ═══════════════════════════════════════════════════════════════════════════════
sep("§1  RISK DISTRIBUTION OVERVIEW")
# ═══════════════════════════════════════════════════════════════════════════════
pooled_inv = pd.concat([obs["p_invasion"], syn["p_invasion"]])

def dist_stats(v, label):
    print(f"\n  {label}  (n = {len(v):,})")
    print(f"    mean   = {v.mean():.4e}")
    print(f"    median = {np.median(v):.4e}")
    print(f"    std    = {v.std():.4e}")
    print(f"    p5     = {np.percentile(v, 5):.4e}")
    print(f"    p90    = {np.percentile(v, 90):.4e}")
    print(f"    p95    = {np.percentile(v, 95):.4e}")
    print(f"    max    = {v.max():.4e}")

dist_stats(obs["p_invasion"], "Observed")
dist_stats(syn["p_invasion"], "Synthetic (Track 2)")
dist_stats(pooled_inv,        f"Pooled (n = {N_OBS + N_SYN:,})")

# ═══════════════════════════════════════════════════════════════════════════════
sep("§2  GAP(i) — TAIL RISK COVERAGE")
# ═══════════════════════════════════════════════════════════════════════════════
p90_obs   = np.percentile(obs["p_invasion"], 90)
p95_obs   = np.percentile(obs["p_invasion"], 95)
p95_syn   = np.percentile(syn["p_invasion"], 95)
tmr_count = (syn["p_invasion"] > p90_obs).sum()
tmr       = (tmr_count / N_SYN) / 0.10
p95_ratio = p95_syn / p95_obs
max_ext   = (syn["p_invasion"].max() / obs["p_invasion"].max() - 1) * 100

print(f"\n  P90 observed          = {p90_obs:.4e}")
print(f"  Synthetic > P90_obs   = {tmr_count:,} / {N_SYN:,}  ({tmr_count/N_SYN*100:.2f}%)")
print(f"  TMR                   = {tmr:.3f}   (threshold ≥ 0.70)  {'✓ PASS' if tmr >= 0.70 else '✗ FAIL'}")
print(f"  P95 obs / syn         = {p95_obs:.4e} / {p95_syn:.4e}")
print(f"  P95 ratio (syn/obs)   = {p95_ratio:.3f}  (threshold ≥ 0.80)  {'✓ PASS' if p95_ratio >= 0.80 else '✗ FAIL'}")
print(f"  Max obs / syn         = {obs['p_invasion'].max():.4e} / {syn['p_invasion'].max():.4e}")
print(f"  Max extension         = {max_ext:+.1f}%  (threshold > 0)  {'✓ PASS' if max_ext > 0 else '✗ FAIL'}")

# ═══════════════════════════════════════════════════════════════════════════════
sep("§3  LAYER 3 VALIDATION")
# ═══════════════════════════════════════════════════════════════════════════════

# 3.1  Fréchet Distance (= KS statistic on CDFs)
ks_overall, ks_p = stats.ks_2samp(obs["p_invasion"], syn["p_invasion"])
print(f"\n  3.1  Fréchet Distance")
print(f"    KS statistic = {ks_overall:.4f}  (threshold < {FRECHET_THRESHOLD})  "
      f"{'✓ PASS' if ks_overall < FRECHET_THRESHOLD else '✗ FAIL'}")
print(f"    KS p-value   = {ks_p:.3e}")

# 3.2  KS by vessel type
print(f"\n  3.2  KS by vessel type")
print(f"    {'Type':<16} {'n_obs':>7} {'n_syn':>7} {'KS':>8} {'p-value':>12}  Result")
print(f"    {'─'*62}")
for vt in sorted(obs["vessel_type"].unique()):
    o = obs[obs["vessel_type"] == vt]["p_invasion"].values
    s = syn[syn["vessel_type"] == vt]["p_invasion"].values
    if len(o) < 10 or len(s) < 10:
        print(f"    {vt:<16} {len(o):>7} {len(s):>7}  (sample too small)")
        continue
    ks, p = stats.ks_2samp(o, s)
    tag = "✓ PASS" if ks < FRECHET_THRESHOLD else ("borderline" if ks < 0.16 else "✗")
    print(f"    {vt:<16} {len(o):>7} {len(s):>7} {ks:>8.3f} {p:>12.2e}  {tag}")

# 3.3  KS by n_ports
print(f"\n  3.3  KS by n_ports")
print(f"    {'n_ports':>7} {'n_obs':>7} {'n_syn':>7} {'KS':>8} {'p-value':>12}  Result")
print(f"    {'─'*54}")
for np_ in sorted(obs["n_ports"].unique()):
    o = obs[obs["n_ports"] == np_]["p_invasion"].values
    s = syn[syn["n_ports"] == np_]["p_invasion"].values
    ks, p = stats.ks_2samp(o, s)
    tag = "✓ PASS" if ks < FRECHET_THRESHOLD else "✗"
    print(f"    {np_:>7} {len(o):>7} {len(s):>7} {ks:>8.3f} {p:>12.2e}  {tag}")

# ═══════════════════════════════════════════════════════════════════════════════
sep("§4  GAP(ii) — MULTI-STOP MONOTONICITY")
# ═══════════════════════════════════════════════════════════════════════════════

def hop_stats_table(hop_df, label):
    print(f"\n  {label}")
    print(f"    {'Hops':>5} {'Mean':>12} {'Median':>12} {'Std':>12} {'Count':>7}")
    print(f"    {'─'*52}")
    means = []
    for h in HOP_RANGE:
        sub = hop_df[hop_df["hops_to_nz"] == h]["pr_inv"]
        if len(sub) == 0:
            continue
        print(f"    {h:>5} {sub.mean():>12.3e} {sub.median():>12.3e} {sub.std():>12.3e} {len(sub):>7,}")
        means.append(sub.mean())
    rho, pv = stats.spearmanr(HOP_RANGE[:len(means)], means)
    print(f"    Spearman ρ = {rho:.3f}  p = {pv:.3e}  "
          f"{'✓ monotone' if all(means[i] > means[i+1] for i in range(len(means)-1)) else '✗ not monotone'}")
    return means

obs_means = hop_stats_table(obs_hop, "4.1  Observed")
syn_means = hop_stats_table(syn_hop, "4.2  Synthetic (Track 2)")

# 4.3  Component breakdown by hop (synthetic)
print(f"\n  4.3  Component breakdown by hop (Synthetic)")
print(f"    {'Hops':>5} {'P_alien':>10} {'P_intro':>10} {'P_estab':>12} {'Pr(Inv)':>12}")
print(f"    {'─'*54}")
for h in HOP_RANGE:
    sub = syn_hop[syn_hop["hops_to_nz"] == h]
    if len(sub) == 0:
        continue
    print(f"    {h:>5} {sub['p_alien'].mean():>10.4f} {sub['p_intro'].mean():>10.4f} "
          f"{sub['p_estab'].mean():>12.3e} {sub['pr_inv'].mean():>12.3e}")

# 4.4  Per-hop KS tests
print(f"\n  4.4  Per-hop KS: Observed vs Synthetic")
print(f"    {'Hops':>5} {'n_obs':>7} {'n_syn':>8} {'KS':>8} {'p-value':>12}  Result")
print(f"    {'─'*54}")
for h in HOP_RANGE:
    o = obs_hop[obs_hop["hops_to_nz"] == h]["pr_inv"].values
    s = syn_hop[syn_hop["hops_to_nz"] == h]["pr_inv"].values
    if len(o) == 0 or len(s) == 0:
        continue
    ks, p = stats.ks_2samp(o, s)
    tag = "✓ PASS" if ks < FRECHET_THRESHOLD else ("borderline" if ks < 0.16 else "✗")
    print(f"    {h:>5} {len(o):>7,} {len(s):>8,} {ks:>8.3f} {p:>12.2e}  {tag}")

# 4.5  Monotonicity tests
print(f"\n  4.5  Monotonicity tests")
for label, hop_df in [("Observed", obs_hop), ("Synthetic", syn_hop)]:
    means_h = [hop_df[hop_df["hops_to_nz"]==h]["pr_inv"].mean() for h in HOP_RANGE
               if len(hop_df[hop_df["hops_to_nz"]==h]) > 0]
    hops_used = [h for h in HOP_RANGE if len(hop_df[hop_df["hops_to_nz"]==h]) > 0]
    rho, pv = stats.spearmanr(hops_used, means_h)
    print(f"    {label:<12}: Spearman ρ = {rho:.3f}  p = {pv:.3e}")

print()
for label, hop_df in [("Observed", obs_hop), ("Synthetic", syn_hop)]:
    h1 = hop_df[hop_df["hops_to_nz"]==1]["pr_inv"].values
    h6 = hop_df[hop_df["hops_to_nz"]==6]["pr_inv"].values
    u, p = stats.mannwhitneyu(h1, h6, alternative="greater")
    print(f"    Mann-Whitney (hops=1 > hops=6) {label}: U = {u:,.0f}  p = {p:.2e}")

# ═══════════════════════════════════════════════════════════════════════════════
sep("§5  NZ DESTINATION DISTRIBUTION & CVAE QC")
# ═══════════════════════════════════════════════════════════════════════════════

# 5.1  Arrival counts
print(f"\n  5.1  Arrival count distribution")
print(f"    {'Port':<16} {'Obs n':>7} {'Obs %':>7} {'Syn n':>7} {'Syn %':>7}")
print(f"    {'─'*46}")
obs_port_n = obs["nz_dest_port"].value_counts()
syn_port_n = syn["nz_dest_port"].value_counts()
all_ports  = sorted(set(obs_port_n.index) | set(syn_port_n.index),
                    key=lambda p: obs_port_n.get(p, 0), reverse=True)
obs_pvec, syn_pvec = [], []
for p in all_ports:
    on = obs_port_n.get(p, 0); sn = syn_port_n.get(p, 0)
    print(f"    {p:<16} {on:>7,} {on/N_OBS*100:>6.1f}% {sn:>7,} {sn/N_SYN*100:>6.1f}%")
    obs_pvec.append(on / N_OBS); syn_pvec.append(sn / N_SYN)

# 5.2  CVAE QC metrics
obs_pvec = np.array(obs_pvec)
syn_pvec = np.array(syn_pvec)
jsd  = jensenshannon(obs_pvec, syn_pvec) ** 2   # returns sqrt(JSD) — square back
js_d = jensenshannon(obs_pvec, syn_pvec)         # JS distance = sqrt(JSD)
tvd  = 0.5 * np.sum(np.abs(obs_pvec - syn_pvec))

print(f"\n  5.2  CVAE QC metrics")
print(f"    JSD (nats)          = {jsd:.5f}")
print(f"    √JSD (JS distance)  = {js_d:.4f}")
print(f"    TVD                 = {tvd:.4f}")
print(f"    Fréchet Distance    = {ks_overall:.4f}  (from §3.1)")

# 5.3  Risk by destination port
print(f"\n  5.3  Risk by NZ destination port")
print(f"    {'Port':<16} {'Obs mean':>12} {'Obs n':>6} {'Syn mean':>12} {'Syn n':>6} {'Δ (%)':>7}")
print(f"    {'─'*62}")
port_risk = []
for p in all_ports:
    om = obs[obs["nz_dest_port"]==p]["p_invasion"].mean()
    sm = syn[syn["nz_dest_port"]==p]["p_invasion"].mean()
    on = obs_port_n.get(p, 0); sn = syn_port_n.get(p, 0)
    dp = (sm/om - 1)*100
    port_risk.append((p, om, on, sm, sn, dp))
    print(f"    {p:<16} {om:>12.4e} {on:>6,} {sm:>12.4e} {sn:>6,} {dp:>+7.1f}%")

# ═══════════════════════════════════════════════════════════════════════════════
sep("§6  RISK BREAKDOWN BY VOYAGE LENGTH")
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n  {'n_ports':>7} {'Obs mean':>12} {'Obs n':>8} {'Syn mean':>12} {'Syn n':>8}")
print(f"  {'─'*52}")
for np_ in sorted(obs["n_ports"].unique()):
    om = obs[obs["n_ports"]==np_]["p_invasion"].mean()
    sm = syn[syn["n_ports"]==np_]["p_invasion"].mean()
    on = (obs["n_ports"]==np_).sum(); sn = (syn["n_ports"]==np_).sum()
    dp = (sm/om - 1)*100
    print(f"  {np_:>7} {om:>12.3e} {on:>7,} ({on/N_OBS*100:.0f}%) {sm:>12.3e} {sn:>7,} ({sn/N_SYN*100:.0f}%)  Δ={dp:+.1f}%")

# ═══════════════════════════════════════════════════════════════════════════════
sep("§7  RISK BREAKDOWN BY VESSEL TYPE")
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n  {'Type':<16} {'Obs mean':>12} {'Obs n':>7} {'Syn mean':>12} {'Syn n':>7} {'Δ (%)':>7} {'KS':>7}")
print(f"  {'─'*70}")
for vt in sorted(obs["vessel_type"].unique()):
    o = obs[obs["vessel_type"]==vt]
    s = syn[syn["vessel_type"]==vt]
    if len(o) < 5 or len(s) < 5:
        continue
    om = o["p_invasion"].mean(); sm = s["p_invasion"].mean()
    dp = (sm/om - 1)*100
    ks, _ = stats.ks_2samp(o["p_invasion"], s["p_invasion"])
    tag = "✓" if ks < FRECHET_THRESHOLD else ("~" if ks < 0.16 else "✗")
    print(f"  {vt:<16} {om:>12.4e} {len(o):>7,} {sm:>12.4e} {len(s):>7,} {dp:>+7.1f}% {ks:>6.3f} {tag}")

# ═══════════════════════════════════════════════════════════════════════════════
sep("§8  POOLED CORPUS SUMMARY")
# ═══════════════════════════════════════════════════════════════════════════════
dist_stats(pooled_inv, f"Pooled (obs + syn, n = {N_OBS+N_SYN:,})")

# ═══════════════════════════════════════════════════════════════════════════════
sep("APPENDIX A — DECAY RATIO")
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n  {'Hops':>5} {'Obs ratio':>11} {'Syn ratio':>11}")
print(f"  {'─'*30}")
obs_h1_mean = obs_hop[obs_hop["hops_to_nz"]==1]["pr_inv"].mean()
syn_h1_mean = syn_hop[syn_hop["hops_to_nz"]==1]["pr_inv"].mean()
for h in HOP_RANGE:
    om = obs_hop[obs_hop["hops_to_nz"]==h]["pr_inv"].mean()
    sm = syn_hop[syn_hop["hops_to_nz"]==h]["pr_inv"].mean()
    print(f"  {h:>5} {om/obs_h1_mean:>11.3f} {sm/syn_h1_mean:>11.3f}")

# ═══════════════════════════════════════════════════════════════════════════════
sep("APPENDIX B — GAP(ii) BY VOYAGE LENGTH (Synthetic)")
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n  {'n_ports':>7}", end="")
for h in HOP_RANGE:
    print(f"  {'hops='+str(h):>12}", end="")
print(f"  {'Monotone':>9}")
print(f"  {'─'*85}")
for np_ in sorted(syn["n_ports"].unique()):
    sub = syn_hop[syn_hop["n_ports"]==np_]
    print(f"  {np_:>7}", end="")
    means_by_hop = []
    for h in HOP_RANGE:
        g = sub[sub["hops_to_nz"]==h]["pr_inv"]
        if len(g) == 0:
            print(f"  {'—':>12}", end="")
        else:
            print(f"  {g.mean():>12.3e}", end="")
            means_by_hop.append(g.mean())
    mono = all(means_by_hop[i] > means_by_hop[i+1] for i in range(len(means_by_hop)-1))
    print(f"  {'✓ YES' if mono else '✗ NO':>9}")

print("\n[Done]")
