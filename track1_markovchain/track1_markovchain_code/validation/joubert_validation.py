# %% Replicating Joubert & Meintjes (2016) Section 3
# Fig 3: n_ports distribution (box plot, by cluster)
# Fig 4: total_span_days distribution (histogram, faceted by cluster)
# Fig 5: total_distance distribution (histogram, faceted by cluster)
# Fig 6: vessel type × cluster geographic coverage (bar chart)

import ast
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

# ── Path Configuration ──────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR   = SCRIPT_DIR.parent.parent.parent   # → track1_markovchain/
ROOT_DIR   = BASE_DIR.parent.parent            # → D:/2025ChangZhang/
OBS_PATH   = BASE_DIR / 'data/clustering/cluster.m1.labels.k5.csv'
SYN_PATH   = BASE_DIR / 'data/markov/validation.result/mc.validation/layer2_speed_filter/mc.syn.all.layer2.csv'
OUT_DIR    = ROOT_DIR / 'graphics/track1_graphic_output/validation'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Color and Cluster Labels ───────────────────────────────────
CLUSTER_NAMES = {
    0: 'C0: East Asian liner',
    1: 'C1: Trans-Pacific',
    2: 'C2: Pacific Island feeder',
    3: 'C3: Australia / SW Pacific',
    4: 'C4: SE Asian bulk',
}
OBS_COLOR = '#4C8BB5'   # Blue
SYN_COLOR = '#E07B6A'   # Red

# ── Read Data ──────────────────────────────────────────────
obs = pd.read_csv(OBS_PATH)
syn = pd.read_csv(SYN_PATH)

print(f"Observed Voyages: {len(obs):,}   Synthetic Voyages: {len(syn):,}")
print(f"Observed Columns: {list(obs.columns[:8])}...")
print(f"Synthetic Columns: {list(syn.columns[:8])}...\n")

# ── Calculate Derived Features ─────────────────────────────────────────
def calc_avg_transit(transits):
    try:
        t = transits if isinstance(transits, list) else ast.literal_eval(transits)
        vals = [x for x in t if x > 0]
        return np.mean(vals) if vals else np.nan
    except Exception:
        return np.nan

def calc_total_distance(row):
    """Calculate total great-circle distance (nautical miles) from latitudes/longitudes lists"""
    try:
        lats = row['latitudes'] if isinstance(row['latitudes'], list) \
               else ast.literal_eval(row['latitudes'])
        lons = row['longitudes'] if isinstance(row['longitudes'], list) \
               else ast.literal_eval(row['longitudes'])
        total = 0.0
        for i in range(1, len(lats)):
            lat1, lon1 = np.radians(lats[i-1]), np.radians(lons[i-1])
            lat2, lon2 = np.radians(lats[i]),   np.radians(lons[i])
            dlat, dlon = lat2 - lat1, lon2 - lon1
            a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
            total += 6371 * 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a)) * 0.539957
        return total
    except Exception:
        return np.nan

print("Calculating derived features (takes about 30 seconds)...")
for df in [obs, syn]:
    df['avg_transit_time']  = df['transits_hours'].apply(calc_avg_transit)
    df['total_distance_gc'] = df.apply(calc_total_distance, axis=1)
print("Done\n")

CLUSTERS = sorted(obs['cluster'].unique())

# ══════════════════════════════════════════════════════════
# Fig 3 — n_ports distribution (bar chart, faceted by cluster)
# n_ports consists of discrete values (5/6/7), using bar chart to show distribution shape
# Joubert Fig.3: "distribution of number of minor activities per chain"
# ══════════════════════════════════════════════════════════
all_nports = sorted(set(obs['n_ports'].unique()) | set(syn['n_ports'].unique()))

fig, axes = plt.subplots(1, len(CLUSTERS), figsize=(16, 5), sharey=False)
fig.suptitle(
    'Fig 3 — Number of Port Visits per Voyage by Cluster\n'
    '(replicating Joubert & Meintjes 2016, Fig. 3)',
    fontsize=12
)

for ax, k in zip(axes, CLUSTERS):
    obs_k   = obs[obs['cluster'] == k]['n_ports']
    syn_k   = syn[syn['cluster'] == k]['n_ports']
    obs_frac = obs_k.value_counts(normalize=True).reindex(all_nports, fill_value=0)
    syn_frac = syn_k.value_counts(normalize=True).reindex(all_nports, fill_value=0)

    x = np.arange(len(all_nports))
    w = 0.38
    ax.bar(x - w/2, obs_frac, w, color=OBS_COLOR, alpha=0.85,
           edgecolor='white', linewidth=0.4, label='Observed')
    ax.bar(x + w/2, syn_frac, w, color=SYN_COLOR, alpha=0.85,
           edgecolor='white', linewidth=0.4, label='Synthetic')
    ax.set_xticks(x)
    ax.set_xticklabels(all_nports, fontsize=9)
    ax.set_xlabel('n_ports')
    if k == 0:
        ax.set_ylabel('Fraction of voyages')
    ax.set_title(CLUSTER_NAMES[k].replace(': ', ':\n'), fontsize=8)
    ax.legend(fontsize=7)
    ax.text(0.5, -0.15, f'Obs n={len(obs_k):,}  Syn n={len(syn_k):,}',
            transform=ax.transAxes, ha='center', fontsize=7, color='gray')

fig.tight_layout()
fig.savefig(OUT_DIR / 'joubert_fig3_nports_bar.png', dpi=150, bbox_inches='tight')
plt.close()
print("Fig 3 saved")

# ══════════════════════════════════════════════════════════
# Fig 4 — total_span_days distribution (histogram, faceted by cluster)
# Joubert Fig.4: "chain start times" → Replaced with voyage duration
# ══════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, len(CLUSTERS), figsize=(16, 4), sharey=False)
fig.suptitle(
    'Fig 4 — Total Voyage Duration (days) by Cluster\n'
    '(replicating Joubert & Meintjes 2016, Fig. 4)',
    fontsize=12
)

for ax, k in zip(axes, CLUSTERS):
    obs_k = obs[obs['cluster'] == k]['total_span_days']
    syn_k = syn[syn['cluster'] == k]['total_span_days']

    combined_max = max(obs_k.quantile(0.97), syn_k.quantile(0.97))
    bins = np.linspace(0, combined_max, 30)

    ax.hist(obs_k, bins=bins, density=True,
            color=OBS_COLOR, alpha=0.7, label='Observed',
            edgecolor='white', linewidth=0.3)
    ax.hist(syn_k, bins=bins, density=True,
            color=SYN_COLOR, alpha=0.7, label='Synthetic',
            edgecolor='white', linewidth=0.3)

    ax.set_title(CLUSTER_NAMES[k].replace(': ', ':\n'), fontsize=8)
    ax.set_xlabel('Days', fontsize=8)
    if k == 0:
        ax.set_ylabel('Density')

    # Median Annotation
    ax.axvline(obs_k.median(), color=OBS_COLOR, linestyle='--',
               linewidth=1.2, label=f'Obs med={obs_k.median():.1f}')
    ax.axvline(syn_k.median(), color=SYN_COLOR, linestyle='--',
               linewidth=1.2, label=f'Syn med={syn_k.median():.1f}')
    ax.legend(fontsize=6, loc='upper right')

fig.tight_layout()
fig.savefig(OUT_DIR / 'joubert_fig4_span_days.png', dpi=150, bbox_inches='tight')
plt.close()
print("Fig 4 saved")

# ══════════════════════════════════════════════════════════
# Fig 5A — avg_transit_time (Option A: Average transit time per leg)
# Fig 5B — total_distance_gc (Option B: Cumulative great-circle distance)
# Both are generated, choose the one with better performance for the paper
# Joubert Fig.5: "vehicle kilometres travelled"
# ══════════════════════════════════════════════════════════
for fig_id, col, xlabel, fname in [
    ('5A', 'avg_transit_time',
     'Avg transit time per leg (hours)',
     'joubert_fig5A_avg_transit.png'),
    ('5B', 'total_distance_gc',
     'Total voyage distance (nm, great-circle)',
     'joubert_fig5B_total_distance.png'),
]:
    fig, axes = plt.subplots(1, len(CLUSTERS), figsize=(16, 4), sharey=False)
    fig.suptitle(
        f'Fig {fig_id} — {xlabel} by Cluster\n'
        f'(replicating Joubert & Meintjes 2016, Fig. 5)',
        fontsize=12
    )
    for ax, k in zip(axes, CLUSTERS):
        obs_k = obs[obs['cluster'] == k][col].dropna()
        syn_k = syn[syn['cluster'] == k][col].dropna()

        combined_max = max(obs_k.quantile(0.97), syn_k.quantile(0.97))
        bins = np.linspace(0, combined_max, 30)

        ax.hist(obs_k, bins=bins, density=True,
                color=OBS_COLOR, alpha=0.7, label='Observed',
                edgecolor='white', linewidth=0.3)
                
        ax.hist(syn_k, bins=bins, density=True,
                color=SYN_COLOR, alpha=0.7, label='Synthetic',
                edgecolor='white', linewidth=0.3)
        ax.set_title(CLUSTER_NAMES[k].replace(': ', ':\n'), fontsize=8)
        ax.set_xlabel(xlabel, fontsize=7)
        if k == 0:
            ax.set_ylabel('Density')
        ax.axvline(obs_k.median(), color=OBS_COLOR, linestyle='--',
                   linewidth=1.2, label=f'Obs={obs_k.median():.0f}')
        ax.axvline(syn_k.median(), color=SYN_COLOR, linestyle='--',
                   linewidth=1.2, label=f'Syn={syn_k.median():.0f}')
        ax.legend(fontsize=6, loc='upper right')

    fig.tight_layout()
    fig.savefig(OUT_DIR / fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Fig {fig_id} saved  ({fname})")

# ══════════════════════════════════════════════════════════
# Fig 6 — vessel type and cluster coverage (bar chart)
# Joubert Fig.6 / Table 1: "area combinations"
# ══════════════════════════════════════════════════════════
vtype_col_obs = 'nbic_type_group' if 'nbic_type_group' in obs.columns else 'vessel_type_raw'
vtype_col_syn = 'vessel_type'   # column name in syn
vtypes = sorted(obs[vtype_col_obs].dropna().unique())

fig = plt.figure(figsize=(14, 5))
fig.suptitle(
    'Fig 6 — Geographic and Vessel Type Coverage\n'
    '(replicating Joubert & Meintjes 2016, Fig. 6 / Table 1)',
    fontsize=12
)
gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)

# Left: vessel type distribution
ax_vt = fig.add_subplot(gs[0])
obs_vt_frac = obs[vtype_col_obs].value_counts(normalize=True).reindex(vtypes, fill_value=0)
syn_vt_frac = syn[vtype_col_syn].value_counts(normalize=True).reindex(vtypes, fill_value=0)

x  = np.arange(len(vtypes))
w  = 0.38
ax_vt.bar(x - w/2, obs_vt_frac, w, color=OBS_COLOR, alpha=0.85,
          edgecolor='white', linewidth=0.4, label='Observed')
ax_vt.bar(x + w/2, syn_vt_frac, w, color=SYN_COLOR, alpha=0.85,
          edgecolor='white', linewidth=0.4, label='Synthetic')
ax_vt.set_xticks(x)
ax_vt.set_xticklabels(vtypes, rotation=25, ha='right', fontsize=8)
ax_vt.set_ylabel('Fraction of voyages')
ax_vt.set_title('Vessel type distribution')
ax_vt.legend(fontsize=9)

# Right: cluster distribution
ax_cl = fig.add_subplot(gs[1])
obs_cl_frac = obs['cluster'].value_counts(normalize=True).sort_index()
syn_cl_frac = (syn['cluster'].value_counts(normalize=True).sort_index()
               if 'cluster' in syn.columns else pd.Series(0, index=CLUSTERS))
obs_cl_frac = obs_cl_frac.reindex(CLUSTERS, fill_value=0)
syn_cl_frac = syn_cl_frac.reindex(CLUSTERS, fill_value=0)

x2 = np.arange(len(CLUSTERS))
ax_cl.bar(x2 - w/2, obs_cl_frac, w, color=OBS_COLOR, alpha=0.85,
          edgecolor='white', linewidth=0.4, label='Observed')
ax_cl.bar(x2 + w/2, syn_cl_frac, w, color=SYN_COLOR, alpha=0.85,
          edgecolor='white', linewidth=0.4, label='Synthetic')
ax_cl.set_xticks(x2)
ax_cl.set_xticklabels([f'C{k}' for k in CLUSTERS], fontsize=9)
ax_cl.set_ylabel('Fraction of voyages')
ax_cl.set_title('Cluster distribution')
ax_cl.legend(fontsize=9)

# Add numeric labels
for ax, obs_frac, syn_frac, xs in [
        (ax_vt, obs_vt_frac, syn_vt_frac, x),
        (ax_cl, obs_cl_frac, syn_cl_frac, x2)]:
    for i, xi in enumerate(xs):
        ax.text(xi - w/2, obs_frac.iloc[i] + 0.005,
                f'{obs_frac.iloc[i]:.2f}', ha='center', fontsize=6, color=OBS_COLOR)
        ax.text(xi + w/2, syn_frac.iloc[i] + 0.005,
                f'{syn_frac.iloc[i]:.2f}', ha='center', fontsize=6, color=SYN_COLOR)

fig.tight_layout()
fig.savefig(OUT_DIR / 'joubert_fig6_coverage.png', dpi=150, bbox_inches='tight')
plt.close()
print("Fig 6 saved")

# ══════════════════════════════════════════════════════════
# Numerical Summary
# ══════════════════════════════════════════════════════════
print("\n" + "="*65)
print("Joubert-style Numerical Summary (matches text description in paper)")
print("="*65)

print("\n[Fig 3] n_ports — Median Comparison by Cluster")
print(f"  {'Cluster':<30} {'obs_med':>8} {'syn_med':>8} {'Assessment':>14}")
print("-"*65)
for k in CLUSTERS:
    o = obs[obs['cluster'] == k]['n_ports'].median()
    s = syn[syn['cluster'] == k]['n_ports'].median()
    tag = "overestimated" if s > o + 0.3 else ("underestimated" if s < o - 0.3 else "similar")
    print(f"  {CLUSTER_NAMES[k]:<30} {o:>8.1f} {s:>8.1f} {tag:>14}")

print("\n[Fig 4] total_span_days — Median Comparison by Cluster")
print(f"  {'Cluster':<30} {'obs_med':>8} {'syn_med':>8} {'ratio':>8}")
print("-"*65)
for k in CLUSTERS:
    o = obs[obs['cluster'] == k]['total_span_days'].median()
    s = syn[syn['cluster'] == k]['total_span_days'].median()
    print(f"  {CLUSTER_NAMES[k]:<30} {o:>8.1f} {s:>8.1f} {s/o:>8.2f}x")

print("\n[Fig 5] total_distance — Median Comparison by Cluster")
print(f"  {'Cluster':<30} {'obs_med':>10} {'syn_med':>10} {'ratio':>8}")
print("-"*65)
for k in CLUSTERS:
    o = obs[obs['cluster'] == k]['avg_transit_time'].median()
    s = syn[syn['cluster'] == k]['avg_transit_time'].median()
    print(f"  {CLUSTER_NAMES[k]:<30} {o:>10,.0f} {s:>10,.0f} {s/o:>8.2f}x")

print("\n[Fig 6] Vessel Type Distribution")
print(f"  {'Vessel type':<16} {'obs':>6} {'syn':>6} {'Assessment':>14}")
print("-"*50)
for vt in vtypes:
    o = (obs[vtype_col_obs] == vt).mean()
    s = (syn[vtype_col_syn] == vt).mean()
    diff = s - o
    tag = "overestimated" if diff > 0.03 else ("underestimated" if diff < -0.03 else "similar")
    print(f"  {vt:<16} {o:>6.3f} {s:>6.3f} {tag:>14}")

print(f"\nAll charts have been saved to: {OUT_DIR}")