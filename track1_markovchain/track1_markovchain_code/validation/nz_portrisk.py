import os
import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path

# ── Path Configuration ──────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR   = SCRIPT_DIR.parent.parent.parent
RISK_PATH = os.path.join(BASE_DIR, "data/seebens/track1_risk_results.csv")
OUT_DIR   = os.path.join(BASE_DIR, "data/seebens")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Read Data ──────────────────────────────────────────────────
print("Reading risk results...")
risk = pd.read_csv(RISK_PATH)
print(f"  Total rows: {len(risk)}")
print(f"  Source values: {risk['source'].unique()}")

# ── Split Observed and Synthetic ────────────────────────────
obs_risk = risk[risk["source"] == "observed"].copy()
syn_risk = risk[risk["source"] == "synthetic"].copy()
print(f"  Observed voyages: {len(obs_risk)}, Synthetic voyages: {len(syn_risk)}")

# ── Statistics Function by NZ Destination Port ────────────────────────
def port_stats(df, label):
    result = (df.groupby("nz_dest_port")["p_invasion"]
                .agg(mean="mean", std="std", n="count")
                .reset_index())
    result.columns = ["nz_dest_port",
                      f"{label}_mean", f"{label}_std", f"{label}_n"]
    return result

obs_stats = port_stats(obs_risk, "obs")
syn_stats = port_stats(syn_risk, "syn")

# ── Merge and Calculate Delta ───────────────────────────────────────────
combined = obs_stats.merge(syn_stats, on="nz_dest_port", how="outer")
combined["delta_pct"] = ((combined["syn_mean"] - combined["obs_mean"])
                          / combined["obs_mean"] * 100).round(1)
combined = combined.sort_values("obs_mean", ascending=False)

# ── KS Test (Per Port) ────────────────────────────────────────
ks_results = []
for port in combined["nz_dest_port"]:
    obs_vals = obs_risk[obs_risk["nz_dest_port"] == port]["p_invasion"]
    syn_vals = syn_risk[syn_risk["nz_dest_port"] == port]["p_invasion"]
    if len(obs_vals) > 0 and len(syn_vals) > 0:
        d, p = stats.ks_2samp(obs_vals, syn_vals)
        ks_results.append({"nz_dest_port": port, "ks_D": round(d, 3), "ks_p": round(p, 4)})
    else:
        ks_results.append({"nz_dest_port": port, "ks_D": None, "ks_p": None})

ks_df = pd.DataFrame(ks_results)
combined = combined.merge(ks_df, on="nz_dest_port", how="left")

# ── Output ──────────────────────────────────────────────────────
out_path = os.path.join(OUT_DIR, "track1_nz_port_risk.csv")
combined.to_csv(out_path, index=False)
print(f"\nResults saved to: {out_path}")
print(combined.to_string(index=False))



import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Visualization ────────────────────────────────────────────────────
def save_fig(fig, path):
    buf = fig.canvas.buffer_rgba()  # Trigger rendering to bypass file lock
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure saved: {path}")

# EP53 Ports (consistent with collaborators)
EP53_PORTS = {"AUCKLAND", "TAURANGA", "GISBORNE"}

# Sort by observed mean in descending order
plot_df = combined.dropna(subset=["obs_mean", "syn_mean"]).copy()
plot_df = plot_df.sort_values("obs_mean", ascending=False).reset_index(drop=True)

ports     = plot_df["nz_dest_port"].tolist()
obs_means = plot_df["obs_mean"].values * 1e4
syn_means = plot_df["syn_mean"].values * 1e4
obs_std   = plot_df["obs_std"].values * 1e4
syn_std   = plot_df["syn_std"].values * 1e4
deltas    = plot_df["delta_pct"].values
n_ports   = len(ports)

x      = np.arange(n_ports)
width  = 0.35

fig, ax = plt.subplots(figsize=(12, 6))

# EP53 Background Shading
ep53_idx = [i for i, p in enumerate(ports) if p.upper() in EP53_PORTS]
if ep53_idx:
    ax.axvspan(min(ep53_idx) - 0.5, max(ep53_idx) + 0.5,
               alpha=0.08, color="steelblue", label="Ecoprovince 53")

# Bar Chart
bars_obs = ax.bar(x - width/2, obs_means, width,
                  color="steelblue", alpha=0.85, label="Observed",
                  yerr=obs_std, capsize=3, error_kw={"elinewidth": 0.8})
bars_syn = ax.bar(x + width/2, syn_means, width,
                  color="tomato", alpha=0.85, label="Synthetic (Track 1)",
                  yerr=syn_std, capsize=3, error_kw={"elinewidth": 0.8})

# Delta Annotation
for i, delta in enumerate(deltas):
    ymax = max(obs_means[i] + obs_std[i], syn_means[i] + syn_std[i])
    sign = "+" if delta >= 0 else ""
    ax.text(x[i], ymax + 0.02, f"{sign}{delta:.1f}%",
            ha="center", va="bottom", fontsize=7.5, color="dimgray")

# Axis Settings
ax.set_xticks(x)
ax.set_xticklabels([p.title() for p in ports], rotation=30, ha="right", fontsize=9)
ax.set_ylabel(r"Mean invasion probability $\bar{P}_j$ ($\times 10^{-4}$)", fontsize=10)
ax.set_title("Track 1 — Mean invasion probability by NZ destination port", fontsize=11)
ax.legend(fontsize=9)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda val, _: f"{val:.2f}"))
ax.set_ylim(0, ax.get_ylim()[1] * 1.15)
ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
fig_path = os.path.join(OUT_DIR, "figures/fig6_nz_port_risk.png")
os.makedirs(os.path.dirname(fig_path), exist_ok=True)
save_fig(fig, fig_path)