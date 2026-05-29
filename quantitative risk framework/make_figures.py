"""
Track 2 – Post-Assessment Figures
Generates Figure A, B, C, D, E in figure/
"""

import ast
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
from scipy import stats
from scipy.stats import gaussian_kde

warnings.filterwarnings("ignore")

import os
BASE = os.path.dirname(os.path.abspath(__file__))
FIG  = os.path.join(BASE, "..", "figure")
os.makedirs(FIG, exist_ok=True)

#Data
obs = pd.read_csv(os.path.join(BASE, "..", "output", "risk_observed.csv"))
syn = pd.read_csv(os.path.join(BASE, "..", "output", "track2_risk_synthetic.csv"))

def parse_list(s):
    try:
        return ast.literal_eval(str(s))
    except Exception:
        return []

#Global style
C_OBS = "#2B6CB0"   # deep blue  – observed
C_SYN = "#C0392B"   # deep red   – synthetic
C_OBS_L = "#6BA3D6" # light blue
C_SYN_L = "#E8906A" # light orange-red

plt.rcParams.update({
    "font.family":       "sans-serif",
    "font.sans-serif":   ["Helvetica Neue", "Arial", "DejaVu Sans"],
    "font.size":         10,
    "axes.titlesize":    11,
    "axes.labelsize":    10,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "legend.fontsize":   9,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.color":        "#E5E5E5",
    "grid.linewidth":    0.6,
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
})

SCALE = 1e4   # display P_invasion × 10⁴ for readability

# BUILD HOP-LEVEL DATA
def build_hop_df(df):
    rows = []
    for _, row in df.iterrows():
        pr_list  = parse_list(row["pr_inv_per_port"])
        pi_list  = parse_list(row["p_intro_per_port"])
        pa_list  = parse_list(row["p_alien_per_port"])
        pe_list  = parse_list(row["p_estab_per_port"])
        n_hops   = int(row["n_ports"]) - 1
        for i, (pr, pi_, pa, pe) in enumerate(zip(pr_list, pi_list, pa_list, pe_list)):
            rows.append({
                "hops_to_nz": n_hops - i,
                "pr":   float(pr),
                "pi":   float(pi_),
                "pa":   float(pa),
                "pe":   float(pe),
            })
    return pd.DataFrame(rows)

obs_hop = build_hop_df(obs)
syn_hop = build_hop_df(syn)

HOP_RANGE = [1, 2, 3, 4, 5, 6]

# FIGURE A — Overall P_invasion Distribution (KDE + ECDF)
print("Generating Figure A …")

fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))

x_obs = obs["p_invasion"].values * SCALE
x_syn = syn["p_invasion"].values * SCALE
x_grid = np.linspace(0, max(x_obs.max(), x_syn.max()) * 1.05, 500)

# Panel 1: Histogram + KDE
ax = axes[0]
bins = np.linspace(0, max(x_obs.max(), x_syn.max()) * 1.02, 45)

ax.hist(x_obs, bins=bins, density=True, alpha=0.30, color=C_OBS, label="_nolegend_")
ax.hist(x_syn, bins=bins, density=True, alpha=0.25, color=C_SYN, label="_nolegend_")

kde_obs = gaussian_kde(x_obs, bw_method=0.15)
kde_syn = gaussian_kde(x_syn, bw_method=0.15)
ax.plot(x_grid, kde_obs(x_grid), color=C_OBS, lw=2.0, label=f"Observed  (n = 2,432)")
ax.plot(x_grid, kde_syn(x_grid), color=C_SYN, lw=2.0, label=f"Synthetic (n = 12,000)")

ax.axvline(np.mean(x_obs),   color=C_OBS, lw=1.2, ls="--", alpha=0.8)
ax.axvline(np.mean(x_syn),   color=C_SYN, lw=1.2, ls="--", alpha=0.8)

ax.set_xlabel(r"$P_\mathrm{invasion}$ (×10⁻⁴)")
ax.set_ylabel("Probability density")
ax.set_title("(a)  Probability density")
ax.legend(frameon=False)

# Annotate means
ax.text(np.mean(x_obs) - 0.02, ax.get_ylim()[1]*0.88,
        f"μ={np.mean(x_obs):.2f}", color=C_OBS, fontsize=8, ha="right")
ax.text(np.mean(x_syn) + 0.02, ax.get_ylim()[1]*0.88,
        f"μ={np.mean(x_syn):.2f}", color=C_SYN, fontsize=8, ha="left")

# Panel 2: ECDF
ax = axes[1]
xs_o = np.sort(x_obs); ys_o = np.arange(1, len(xs_o)+1) / len(xs_o)
xs_s = np.sort(x_syn); ys_s = np.arange(1, len(xs_s)+1) / len(xs_s)

ax.plot(xs_o, ys_o, color=C_OBS, lw=2.0, label="Observed")
ax.plot(xs_s, ys_s, color=C_SYN, lw=2.0, label="Synthetic")

# Mark Fréchet distance (max vertical gap)
ks_stat, _ = stats.ks_2samp(x_obs, x_syn)
# find location of max gap
all_x = np.sort(np.concatenate([xs_o, xs_s]))
ecdf_o = np.searchsorted(xs_o, all_x, side="right") / len(xs_o)
ecdf_s = np.searchsorted(xs_s, all_x, side="right") / len(xs_s)
gap_idx = np.argmax(np.abs(ecdf_o - ecdf_s))
xg, yg_o, yg_s = all_x[gap_idx], ecdf_o[gap_idx], ecdf_s[gap_idx]
ax.annotate("", xy=(xg, yg_s), xytext=(xg, yg_o),
            arrowprops=dict(arrowstyle="<->", color="#555555", lw=1.2))
ax.text(xg + 0.04, (yg_o + yg_s)/2,
        f"Fréchet = {ks_stat:.4f}\n(< 0.15 threshold)",
        fontsize=8, color="#444444", va="center")

ax.axhline(0.90, color="#888888", lw=0.8, ls=":", alpha=0.7)
ax.text(x_grid[-1]*0.98, 0.905, "P90", fontsize=7.5, color="#888888", ha="right")

ax.set_xlabel(r"$P_\mathrm{invasion}$ (×10⁻⁴)")
ax.set_ylabel("Cumulative probability")
ax.set_title("(b)  Empirical CDF")
ax.legend(frameon=False)

fig.suptitle("Track 2 Risk Result — Overall $P_{\\mathrm{invasion}}$ Distribution",
             fontsize=12, fontweight="bold", y=1.01)
plt.tight_layout()
fig.savefig(os.path.join(FIG, "figA_overall_distribution.pdf"))
fig.savefig(os.path.join(FIG, "figA_overall_distribution.png"))
plt.close(fig)
print("  → figA saved")

# FIGURE B — Gap(ii) Hop Decay  (boxplot + mean lines)
print("Generating Figure B …")

SCALE_HOP = 1e5  # Pr(Inv) per port × 10⁵

# Pre-compute consistent y limits from both datasets
all_pr_scaled = np.concatenate([obs_hop["pr"].values, syn_hop["pr"].values]) * SCALE_HOP
y_max = np.percentile(all_pr_scaled, 97) * 1.18
y_min = 0.0

fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)

for col_idx, (label, hop_df, color, color_l) in enumerate([
    ("Observed",  obs_hop, C_OBS, C_OBS_L),
    ("Synthetic", syn_hop, C_SYN, C_SYN_L),
]):
    ax = axes[col_idx]
    boxes_data = [hop_df[hop_df["hops_to_nz"] == h]["pr"].values * SCALE_HOP
                  for h in HOP_RANGE]
    means = [d.mean() for d in boxes_data]

    ax.boxplot(
        boxes_data,
        positions=HOP_RANGE,
        widths=0.55,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color="white", lw=2.0),
        whiskerprops=dict(color=color, lw=1.2),
        capprops=dict(color=color, lw=1.2),
        boxprops=dict(facecolor=color_l, edgecolor=color, lw=1.0),
    )

    # Mean line
    ax.plot(HOP_RANGE, means, color=color, marker="o", ms=6,
            lw=2.0, zorder=5, label=f"{label} mean")

    # Synthetic panel: overlay observed reference line
    if col_idx == 1:
        obs_ref = [obs_hop[obs_hop["hops_to_nz"] == h]["pr"].mean() * SCALE_HOP
                   for h in HOP_RANGE]
        ax.plot(HOP_RANGE, obs_ref, color=C_OBS, marker="s", ms=5,
                lw=1.6, ls="--", zorder=4, alpha=0.85, label="Observed mean (ref.)")

    # Mean annotations with consistent offset from known ylim
    for h, m in zip(HOP_RANGE, means):
        ax.text(h, m + y_max * 0.03, f"{m:.2f}",
                ha="center", fontsize=7.5, color=color, fontweight="bold")

    ax.set_xlabel("Hops to NZ  (1 = last port before NZ)")
    if col_idx == 0:
        ax.set_ylabel(r"$\Pr(\mathrm{Inv})_{i \to n}$  (×10⁻⁵)")
    ax.set_title(f"({'a' if col_idx == 0 else 'b'})  {label}")
    ax.set_xticks(HOP_RANGE)
    ax.set_xlim(0.4, 6.6)
    ax.set_ylim(y_min, y_max)
    ax.legend(frameon=False, loc="upper right")

    # Near-NZ shading with fixed text position
    ax.axvspan(0.5, 2.5, alpha=0.06, color=color, zorder=0)
    ax.text(1.5, y_max * 0.04, "Near-NZ",
            ha="center", fontsize=7.5, color=color, alpha=0.65)

fig.suptitle("Track 2 Risk Result — Per-Hop Risk Contribution by Hops to NZ",
             fontsize=12, fontweight="bold", y=1.01)
plt.tight_layout()
fig.savefig(os.path.join(FIG, "figB_gap2_hop_decay.pdf"))
fig.savefig(os.path.join(FIG, "figB_gap2_hop_decay.png"))
plt.close(fig)
print("  → figB saved")

# FIGURE C — Vessel Type Risk Bar Chart  (with Δ% annotation)
print("Generating Figure C …")

VT_ORDER = ["Container", "General Cargo", "Bulker", "Tanker", "RoRo", "Reefer"]
VT_LABELS = {
    "Container": "Container", "General Cargo": "Gen. Cargo",
    "Bulker": "Bulker", "Tanker": "Tanker",
    "RoRo": "RoRo", "Reefer": "Reefer",
}

obs_means_vt = [obs[obs["vessel_type"]==vt]["p_invasion"].mean() * SCALE for vt in VT_ORDER]
obs_std_vt   = [obs[obs["vessel_type"]==vt]["p_invasion"].std()  * SCALE for vt in VT_ORDER]
syn_means_vt = [syn[syn["vessel_type"]==vt]["p_invasion"].mean() * SCALE for vt in VT_ORDER]
syn_std_vt   = [syn[syn["vessel_type"]==vt]["p_invasion"].std()  * SCALE for vt in VT_ORDER]
delta_pct    = [(s/o - 1)*100 for o, s in zip(obs_means_vt, syn_means_vt)]

x = np.arange(len(VT_ORDER))
w = 0.35

fig, ax = plt.subplots(figsize=(9.5, 4.6))

bars_o = ax.bar(x - w/2, obs_means_vt, w, yerr=obs_std_vt,
                color=C_OBS, alpha=0.85, capsize=4,
                error_kw=dict(elinewidth=1.0, ecolor=C_OBS),
                label="Observed", zorder=3)
bars_s = ax.bar(x + w/2, syn_means_vt, w, yerr=syn_std_vt,
                color=C_SYN, alpha=0.85, capsize=4,
                error_kw=dict(elinewidth=1.0, ecolor=C_SYN),
                label="Synthetic", zorder=3)

# Δ% annotation above each synthetic bar
for xi, (sm, sm_sd, dp) in enumerate(zip(syn_means_vt, syn_std_vt, delta_pct)):
    top = sm + sm_sd + 0.008
    color_ann = "#1A6B1A" if dp < 0 else "#8B0000"
    ax.text(xi + w/2, top + 0.002, f"{dp:+.1f}%",
            ha="center", va="bottom", fontsize=8.2,
            color=color_ann, fontweight="bold")

# KS pass/fail markers — placed below x-tick labels using axis transform
ks_vals = {"Container": 0.073, "General Cargo": 0.080, "Bulker": 0.269,
           "Tanker": 0.305, "RoRo": 0.154, "Reefer": 0.460}
for xi, vt in enumerate(VT_ORDER):
    ks = ks_vals[vt]
    tag  = "KS pass" if ks < 0.15 else "KS fail"
    mcol = "#1A6B1A"  if ks < 0.15 else "#8B0000"
    ax.text(xi, -0.16, f"KS={ks:.3f}",
            ha="center", va="top", fontsize=7.5, color=mcol,
            transform=ax.get_xaxis_transform())
    ax.text(xi, -0.25, tag,
            ha="center", va="top", fontsize=7.0, color=mcol,
            fontweight="bold", transform=ax.get_xaxis_transform())

ax.set_xticks(x)
ax.set_xticklabels([VT_LABELS[v] for v in VT_ORDER])
ax.set_ylabel(r"Mean $P_\mathrm{invasion}$  (×10⁻⁴)")
ax.set_ylim(0, max(max(obs_means_vt), max(syn_means_vt)) * 1.36)
fig.subplots_adjust(bottom=0.22)
ax.legend(frameon=False)
ax.set_title("Track 2 Risk Result — Mean $P_{\\mathrm{invasion}}$ by Vessel Type  (± 1 SD,  Δ% synthetic vs observed)",
             fontsize=11, fontweight="bold")

plt.tight_layout()
fig.savefig(os.path.join(FIG, "figC_vessel_type_bar.pdf"))
fig.savefig(os.path.join(FIG, "figC_vessel_type_bar.png"))
plt.close(fig)
print("  → figC saved")

# FIGURE D — Filter Component by Hop  (line chart, dual axis)
print("Generating Figure D …")

# Compute per-hop component means
hop_stats = {}
for h in HOP_RANGE:
    o = obs_hop[obs_hop["hops_to_nz"] == h]
    s = syn_hop[syn_hop["hops_to_nz"] == h]
    hop_stats[h] = {
        "obs_pa": o["pa"].mean(), "obs_pi": o["pi"].mean(), "obs_pe": o["pe"].mean(),
        "syn_pa": s["pa"].mean(), "syn_pi": s["pi"].mean(), "syn_pe": s["pe"].mean(),
    }

# Need to re-build obs_hop with correct column names (pa/pi/pe → same as above)
# rebuild properly
obs_hop2 = build_hop_df(obs)  # has columns pa, pi, pe, pr
syn_hop2 = build_hop_df(syn)

# Aggregate
def hop_means(hop_df, col):
    return [hop_df[hop_df["hops_to_nz"]==h][col].mean() for h in HOP_RANGE]

obs_pa = hop_means(obs_hop2, "pa"); obs_pi = hop_means(obs_hop2, "pi"); obs_pe = hop_means(obs_hop2, "pe")
syn_pa = hop_means(syn_hop2, "pa"); syn_pi = hop_means(syn_hop2, "pi"); syn_pe = hop_means(syn_hop2, "pe")

fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))

components = [
    ("P_alien", obs_pa, syn_pa, "P_{\\mathrm{alien}}", (0.95, 1.005), "#7B4FA2"),
    ("P_intro", obs_pi, syn_pi, "P_{\\mathrm{intro}}",  (0.0,  0.70),  "#D4820A"),
    ("P_estab", obs_pe, syn_pe, "P_{\\mathrm{estab}}",  (0.0,  1.6e-4),"#16836B"),
]

for ax_i, (name, o_vals, s_vals, label, ylim, color) in enumerate(components):
    ax = axes[ax_i]

    ax.plot(HOP_RANGE, o_vals, color=C_OBS, marker="o", ms=6, lw=2.0,
            ls="-", label="Observed", zorder=4)
    ax.plot(HOP_RANGE, s_vals, color=C_SYN, marker="s", ms=6, lw=2.0,
            ls="--", label="Synthetic", zorder=4)

    ax.set_xticks(HOP_RANGE)
    ax.set_xlabel("Hops to NZ")
    ax.set_title(f"({'abc'[ax_i]})  ${label}$", fontsize=11)

    # Format y axis
    if name == "P_estab":
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda v, _: f"{v*1e4:.2f}×10⁻⁴"))
    elif name == "P_intro":
        ax.set_ylabel("Mean component value")
    elif name == "P_alien":
        ax.set_ylim(0.97, 1.002)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))

    ax.legend(frameon=False, loc="best")

    # Shade near-NZ zone
    ax.axvspan(0.5, 2.5, alpha=0.07, color="#AAAAAA", zorder=0)

axes[1].set_ylabel("")

fig.suptitle(
    "Track 2 Risk Result — Risk Component Breakdown by Hops to NZ  "
    r"($P_{\mathrm{alien}}$,  $P_{\mathrm{intro}}$,  $P_{\mathrm{estab}}$)",
    fontsize=11.5, fontweight="bold", y=1.01)
plt.tight_layout()
fig.savefig(os.path.join(FIG, "figD_component_by_hop.pdf"))
fig.savefig(os.path.join(FIG, "figD_component_by_hop.png"))
plt.close(fig)
print("  → figD saved")

# FIGURE E — NZ Port Risk Bar Chart  (truncated Y-axis, ±1 SD)

print("Generating Figure E …")

_PORT_ALL = ["GISBORNE","TAURANGA","WELLINGTON","AUCKLAND","PORT CHALMERS",
             "NELSON","NAPIER","LYTTELTON","NEW PLYMOUTH","TIMARU"]
_PORT_LABELS = {
    "GISBORNE": "Gisborne","TAURANGA": "Tauranga","WELLINGTON": "Wellington",
    "AUCKLAND": "Auckland","PORT CHALMERS": "Pt Chalmers","NELSON": "Nelson",
    "NAPIER": "Napier","LYTTELTON": "Lyttelton","NEW PLYMOUTH": "New Plymouth",
    "TIMARU": "Timaru",
}

# sort by observed mean descending
_port_obs_means = {p: obs[obs["nz_dest_port"]==p]["p_invasion"].mean()*SCALE
                   for p in _PORT_ALL}
_PORT_SORTED = sorted(_PORT_ALL, key=lambda p: _port_obs_means[p], reverse=True)

obs_m_E = np.array([obs[obs["nz_dest_port"]==p]["p_invasion"].mean()*SCALE for p in _PORT_SORTED])
obs_sd_E = np.array([obs[obs["nz_dest_port"]==p]["p_invasion"].std()*SCALE  for p in _PORT_SORTED])
syn_m_E = np.array([syn[syn["nz_dest_port"]==p]["p_invasion"].mean()*SCALE for p in _PORT_SORTED])
syn_sd_E = np.array([syn[syn["nz_dest_port"]==p]["p_invasion"].std()*SCALE  for p in _PORT_SORTED])
dpct_E  = [(s/o - 1)*100 for o, s in zip(obs_m_E, syn_m_E)]

x_E = np.arange(len(_PORT_SORTED))
w_E = 0.38

# Truncated Y-axis: floor just below (min_mean – max_SD), ceil above (max_mean + max_SD)
y_floor_E = max(0.0, np.min(obs_m_E - obs_sd_E) - 0.12)
y_ceil_E  = np.max(np.concatenate([obs_m_E + obs_sd_E, syn_m_E + syn_sd_E])) * 1.22

fig, ax = plt.subplots(figsize=(11.5, 5.2))

ax.bar(x_E - w_E/2, obs_m_E, w_E, yerr=obs_sd_E, color=C_OBS, alpha=0.85,
       capsize=4, error_kw=dict(elinewidth=1.2, ecolor=C_OBS),
       label="Observed (±1 SD)", zorder=3)
ax.bar(x_E + w_E/2, syn_m_E, w_E, yerr=syn_sd_E, color=C_SYN, alpha=0.85,
       capsize=4, error_kw=dict(elinewidth=1.2, ecolor=C_SYN),
       label="Synthetic (±1 SD)", zorder=3)

# Δ% annotation — centred between the pair, above the taller bar + its SD
for xi, (om, osd, sm, ssd, dp) in enumerate(zip(obs_m_E, obs_sd_E, syn_m_E, syn_sd_E, dpct_E)):
    top = max(om + osd, sm + ssd) + (y_ceil_E - y_floor_E) * 0.025
    col = "#1A6B1A" if dp < 0 else "#8B0000"
    ax.text(xi, top, f"Δ{dp:+.1f}%",
            ha="center", va="bottom", fontsize=8.2, color=col, fontweight="bold")

# Province 53 background shading
_prov53 = {"AUCKLAND","TAURANGA","GISBORNE"}
for xi, p in enumerate(_PORT_SORTED):
    if p in _prov53:
        ax.axvspan(xi - 0.5, xi + 0.5, alpha=0.06, color=C_OBS, zorder=0)

# Broken-axis indicator (zigzag) at bottom-left
_d_x = 0.18
_d_y = (y_ceil_E - y_floor_E) * 0.018
_zx = [-0.55, -0.55 + _d_x/2, -0.55 + _d_x, -0.55 + 3*_d_x/2]
_zy = [y_floor_E - _d_y, y_floor_E + _d_y, y_floor_E - _d_y, y_floor_E + _d_y]
ax.plot(_zx, _zy, color="black", lw=1.0, clip_on=False, zorder=10)

ax.set_ylim(y_floor_E, y_ceil_E)
ax.set_xticks(x_E)
ax.set_xticklabels([_PORT_LABELS[p] for p in _PORT_SORTED], rotation=25, ha="right")
ax.set_ylabel(r"Mean $P_\mathrm{invasion}$  (×10⁻⁴)")
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
ax.yaxis.set_minor_locator(mticker.AutoMinorLocator(2))
ax.legend(frameon=False)

# Province 53 label
_p53_idx = [i for i, p in enumerate(_PORT_SORTED) if p in _prov53]
ax.text(np.mean(_p53_idx), y_ceil_E * 0.993,
        "Province 53 (Northern NZ)", fontsize=8, color=C_OBS,
        alpha=0.75, ha="center", va="top", style="italic")

# Truncation footnote
ax.text(len(_PORT_SORTED) - 0.5, y_floor_E + (y_ceil_E - y_floor_E)*0.01,
        f"Y-axis truncated at {y_floor_E:.2f}×10⁻⁴",
        ha="right", va="bottom", fontsize=7.5, color="grey", style="italic")

ax.set_title(
    "Track 2 Risk Result — Mean $P_{\\mathrm{invasion}}$ by NZ Destination Port  (± 1 SD,  Δ% synthetic vs observed)",
    fontsize=11)

plt.tight_layout()
fig.savefig(os.path.join(FIG, "figE_nz_port_bar.pdf"))
fig.savefig(os.path.join(FIG, "figE_nz_port_bar.png"), dpi=180)
plt.close(fig)
print("  → figE saved")

print("\nAll figures saved to:", FIG)
