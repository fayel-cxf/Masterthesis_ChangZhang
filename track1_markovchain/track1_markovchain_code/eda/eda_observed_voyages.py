"""
eda_observed_voyages.py
========================
Track 1 — Exploratory Data Analysis of Observed Voyages

Input:  data/clustering/cluster.m1.labels.k5.csv  (or observed_voyages_cleaned.csv)
Output: graphics/eda/ — all figures listed below

Figures:
  fig01_fleet_overview.png          — fleet overview (vessel type, DWT)
  fig02_voyage_structure.png        — voyage structure (n_ports, total span)
  fig03_transit_distribution.png    — per-leg transit time distribution
  fig04_dwell_distribution.png      — intermediate port dwell time distribution
  fig05_transit_by_nports.png       — transit time grouped by n_ports
  fig06_transit_by_vtype.png        — transit time grouped by vessel type
  fig07_port_network.png            — port visit frequency
  fig08_nz_destination.png          — New Zealand destination port distribution
  fig09_origin_geography.png        — origin port geographic distribution (lat/lon heatmap)
  fig10_ecoprovince.png             — ecoprovince coverage
  fig11_temporal.png                — temporal analysis
  fig12_dwt_by_vtype.png            — DWT distribution by vessel type
  fig13_correlation.png             — correlation matrix of key variables
  fig14_delta_t.png                 — Δt (cumulative transit+dwell from origin to NZ)
"""

import ast
import re
import warnings
from pathlib import Path
from math import radians, sin, cos, sqrt, atan2

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import gaussian_kde, ks_2samp

warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════
# 0. Path configuration
# ══════════════════════════════════════════════════════════

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR   = SCRIPT_DIR.parent.parent.parent   # → track1_markovchain/
DATA_PATH  = BASE_DIR / 'data/clustering/cluster.m1.labels.k5.csv'
OUT_DIR    = BASE_DIR / 'graphics/eda'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Colour scheme
PALETTE = ['#2E86AB', '#E84855', '#3BB273', '#F4A261',
           '#9B5DE5', '#F15BB5', '#00BBF9', '#FEE440']
VTYPE_COLORS = {
    'Bulker':        '#2E86AB',
    'Container':     '#E84855',
    'General Cargo': '#3BB273',
    'Tanker':        '#F4A261',
    'RoRo':          '#9B5DE5',
    'Reefer':        '#F15BB5',
    'Passenger':     '#00BBF9',
    'Other':         '#FEE440',
}
C_GRID = '#E8E8E8'

# ══════════════════════════════════════════════════════════
# 1. Data loading
# ══════════════════════════════════════════════════════════

def safe_eval(val):
    """Parse list-typed fields from CSV strings."""
    if pd.isna(val): return []
    s = re.sub(r'np\.\w+\(([^)]+)\)', r'\1', str(val))
    try: return ast.literal_eval(s)
    except: return []

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1r, lat2r = radians(lat1), radians(lat2)
    dlat = lat2r - lat1r
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(lat1r)*cos(lat2r)*sin(dlon/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1-a))

print('[Load] Reading data...')
df = pd.read_csv(DATA_PATH)
LIST_COLS = ['ports', 'port_territories', 'latitudes', 'longitudes',
             'ecoprovinces', 'dwells_hours', 'transits_hours']
for col in LIST_COLS:
    if col in df.columns:
        df[col] = df[col].apply(safe_eval)

if 'voyage_start_dt' in df.columns:
    df['voyage_start_dt'] = pd.to_datetime(df['voyage_start_dt'], errors='coerce')
    df['voyage_end_dt']   = pd.to_datetime(df['voyage_end_dt'],   errors='coerce')
    df['start_month'] = df['voyage_start_dt'].dt.month
    df['start_year']  = df['voyage_start_dt'].dt.year

# Compute delta_t: cumulative transit + dwell from origin port to NZ
def compute_delta_t(row):
    trans  = row['transits_hours']
    dwells = row['dwells_hours']
    n = len(trans)
    if n < 2: return 0.0
    return sum(trans[1:]) + sum(dwells[1:n-1])

df['delta_t_hours'] = df.apply(compute_delta_t, axis=1)

# Expand per-leg records
leg_records = []
for _, row in df.iterrows():
    ports  = row['ports']
    trans  = row['transits_hours']
    dwells = row['dwells_hours']
    lats   = row['latitudes']
    lons   = row['longitudes']
    vtype  = row['nbic_type_group']
    n      = len(ports)
    ngroup = 'n=5' if n == 5 else ('n=6' if n == 6 else 'n=7')

    for i in range(1, n):
        t = trans[i] if i < len(trans) else None
        if not t or t <= 0: continue
        try:
            dist = haversine_km(lats[i-1], lons[i-1], lats[i], lons[i])
            speed = (dist * 0.5399) / t   # convert km/h → knots
        except Exception:
            dist  = None
            speed = None
        leg_records.append({
            'vtype':     vtype,
            'transit_h': t,
            'leg_idx':   i,
            'n_ports':   n,
            'ngroup':    ngroup,
            'dist_km':   dist,
            'speed_kt':  speed if speed and 2 < speed < 35 else None,
        })

legs_df = pd.DataFrame(leg_records)

# Expand dwell records (intermediate ports only — exclude origin and NZ destination)
dwell_records = []
for _, row in df.iterrows():
    ports  = row['ports']
    dwells = row['dwells_hours']
    vtype  = row['nbic_type_group']
    n      = len(ports)
    for i in range(1, n-1):
        d = dwells[i] if i < len(dwells) else None
        if d and d > 0:
            dwell_records.append({'vtype': vtype, 'dwell_h': d, 'n_ports': n})

dwells_df = pd.DataFrame(dwell_records)

print(f'       {len(df)} voyages, {len(legs_df)} legs, {len(dwells_df)} dwell records')
vtypes = sorted(df['nbic_type_group'].unique())

# ══════════════════════════════════════════════════════════
# Helper functions
# ══════════════════════════════════════════════════════════

def style_ax(ax, title='', xlabel='', ylabel='', fontsize=9):
    ax.set_facecolor('#FAFAFA')
    ax.grid(True, color=C_GRID, linewidth=0.7, zorder=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#CCCCCC')
    ax.spines['bottom'].set_color('#CCCCCC')
    if title:  ax.set_title(title, fontsize=fontsize, fontweight='bold', pad=6)
    if xlabel: ax.set_xlabel(xlabel, fontsize=fontsize-1)
    if ylabel: ax.set_ylabel(ylabel, fontsize=fontsize-1)
    ax.tick_params(labelsize=fontsize-2)

def savefig(fig, name):
    path = OUT_DIR / name
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  Saved: {name}')

# ══════════════════════════════════════════════════════════
# Fig 01 — Fleet overview
# ══════════════════════════════════════════════════════════

print('[Fig 01] Fleet overview...')
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.patch.set_facecolor('white')
fig.suptitle('Fleet Overview', fontsize=13, fontweight='bold')

# Vessel type distribution (pie chart)
ax = axes[0]
counts = df['nbic_type_group'].value_counts()
colors = [VTYPE_COLORS.get(v, '#AAAAAA') for v in counts.index]
wedges, texts, autotexts = ax.pie(
    counts.values, labels=counts.index, autopct='%1.1f%%',
    colors=colors, startangle=90, pctdistance=0.75,
    textprops={'fontsize': 8}
)
for at in autotexts:
    at.set_fontsize(7)
style_ax(ax, f'Vessel Type Distribution\n(n={len(df):,} voyages)')

# DWT distribution (global KDE)
ax = axes[1]
dwt = df['DWT'].values
xs = np.linspace(dwt.min(), dwt.max(), 400)
kde = gaussian_kde(dwt, bw_method=0.3)
ax.plot(xs/1e4, kde(xs)*1e4, color='#2E86AB', lw=2)
ax.fill_between(xs/1e4, kde(xs)*1e4, alpha=0.3, color='#2E86AB')
ax.axvline(np.median(dwt)/1e4, color='#E84855', lw=1.5, ls='--',
           label=f'Median={np.median(dwt)/1e3:.0f}k DWT')
style_ax(ax, 'DWT Distribution', 'DWT (×10⁴ tonnes)', 'Density')
ax.legend(fontsize=8)

# DWT boxplot by vessel type
ax = axes[2]
data = [df[df['nbic_type_group'] == vt]['DWT'].values / 1e3 for vt in vtypes]
bp = ax.boxplot(data, patch_artist=True,
                medianprops=dict(color='white', lw=2),
                flierprops=dict(marker='.', markersize=2, alpha=0.4))
for patch, vt in zip(bp['boxes'], vtypes):
    patch.set_facecolor(VTYPE_COLORS.get(vt, '#AAAAAA'))
    patch.set_alpha(0.7)
ax.set_xticklabels(vtypes, rotation=30, ha='right', fontsize=7)
style_ax(ax, 'DWT by Vessel Type', 'Vessel Type', 'DWT (× 10³ tonnes)')

plt.tight_layout()
savefig(fig, 'fig01_fleet_overview.png')

# ══════════════════════════════════════════════════════════
# Fig 02 — Voyage structure
# ══════════════════════════════════════════════════════════

print('[Fig 02] Voyage structure...')
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.patch.set_facecolor('white')
fig.suptitle('Voyage Structure Overview', fontsize=13, fontweight='bold')

# n_ports distribution
ax = axes[0]
nc = df['n_ports'].value_counts().sort_index()
bars = ax.bar(nc.index.astype(str), nc.values,
              color=['#2E86AB', '#E84855', '#3BB273'], alpha=0.8)
for bar, val in zip(bars, nc.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
            f'{val:,}\n({val/len(df)*100:.1f}%)', ha='center', fontsize=8)
style_ax(ax, 'Number of Ports Distribution', 'n_ports', 'Count')

# Total span distribution by n_ports (KDE)
ax = axes[1]
for n, color in [(5, '#2E86AB'), (6, '#E84855'), (7, '#3BB273')]:
    sub = df[df['n_ports'] == n]['total_span_days'].values
    if len(sub) > 5:
        xs = np.linspace(sub.min(), sub.max(), 300)
        kde = gaussian_kde(sub, bw_method=0.3)
        ax.plot(xs, kde(xs), color=color, lw=2, label=f'n={n} (n={len(sub):,})')
        ax.fill_between(xs, kde(xs), alpha=0.2, color=color)
style_ax(ax, 'Voyage Span by n_ports', 'Total span (days)', 'Density')
ax.legend(fontsize=8)

# Total span vs n_ports scatter (jittered)
ax = axes[2]
for n, color in [(5, '#2E86AB'), (6, '#E84855'), (7, '#3BB273')]:
    sub = df[df['n_ports'] == n]
    ax.scatter(sub['n_ports'] + np.random.uniform(-0.1, 0.1, len(sub)),
               sub['total_span_days'], color=color, alpha=0.3, s=8, label=f'n={n}')
ax.set_xticks([5, 6, 7])
style_ax(ax, 'Span Days vs n_ports', 'n_ports', 'Total span (days)')
ax.legend(fontsize=8)

plt.tight_layout()
savefig(fig, 'fig02_voyage_structure.png')

# ══════════════════════════════════════════════════════════
# Fig 03 — Transit time distribution
# ══════════════════════════════════════════════════════════

print('[Fig 03] Transit time distribution...')
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.patch.set_facecolor('white')
fig.suptitle('Per-Leg Transit Time Distribution', fontsize=13, fontweight='bold')

clip = 500
tv = legs_df[legs_df['transit_h'] <= clip]['transit_h'].values

# KDE
ax = axes[0]
xs = np.linspace(0, clip, 400)
kde = gaussian_kde(tv, bw_method=0.2)
ax.plot(xs, kde(xs), color='#2E86AB', lw=2)
ax.fill_between(xs, kde(xs), alpha=0.3, color='#2E86AB')
ax.axvline(np.median(tv), color='#E84855', lw=1.5, ls='--',
           label=f'Median={np.median(tv):.0f}h')
ax.axvline(np.mean(tv), color='#3BB273', lw=1.5, ls=':',
           label=f'Mean={np.mean(tv):.0f}h')
style_ax(ax, 'Overall Transit KDE (≤500h)', 'Transit time (hours)', 'Density')
ax.legend(fontsize=8)

# CDF
ax = axes[1]
sorted_tv = np.sort(tv)
cdf = np.arange(1, len(sorted_tv)+1) / len(sorted_tv)
ax.plot(sorted_tv, cdf, color='#2E86AB', lw=2)
ax.axvline(72,  color='#E84855', lw=1, ls='--', alpha=0.7, label='72h (3 days)')
ax.axvline(168, color='#3BB273', lw=1, ls='--', alpha=0.7, label='168h (7 days)')
ax.axvline(336, color='#F4A261', lw=1, ls='--', alpha=0.7, label='336h (14 days)')
style_ax(ax, 'Transit Time CDF', 'Transit time (hours)', 'Cumulative Probability')
ax.legend(fontsize=8)

# Percentile summary
ax = axes[2]
pcts = [10, 25, 50, 75, 90, 95, 99]
vals = [np.percentile(tv, p) for p in pcts]
ax.barh([f'P{p}' for p in pcts], vals, color='#2E86AB', alpha=0.7)
for i, (p, v) in enumerate(zip(pcts, vals)):
    ax.text(v + 2, i, f'{v:.0f}h', va='center', fontsize=8)
style_ax(ax, 'Percentile Summary', 'Transit time (hours)', '')

plt.tight_layout()
savefig(fig, 'fig03_transit_distribution.png')

# ══════════════════════════════════════════════════════════
# Fig 04 — Dwell time distribution
# ══════════════════════════════════════════════════════════

print('[Fig 04] Dwell time distribution...')
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.patch.set_facecolor('white')
fig.suptitle('Intermediate Port Dwell Time Distribution', fontsize=13, fontweight='bold')

dv = dwells_df['dwell_h'].values
dv_clip = dv[dv <= 200]

# KDE
ax = axes[0]
xs = np.linspace(0, 200, 400)
kde = gaussian_kde(dv_clip, bw_method=0.2)
ax.plot(xs, kde(xs), color='#3BB273', lw=2)
ax.fill_between(xs, kde(xs), alpha=0.3, color='#3BB273')
ax.axvline(np.median(dv_clip), color='#E84855', lw=1.5, ls='--',
           label=f'Median={np.median(dv_clip):.0f}h')
ax.axvline(6,   color='#888888', lw=1, ls=':', alpha=0.7, label='Min bound (6h)')
ax.axvline(136, color='#888888', lw=1, ls=':', alpha=0.7, label='Max bound (136h)')
style_ax(ax, 'Dwell Time KDE (≤200h)', 'Dwell time (hours)', 'Density')
ax.legend(fontsize=8)

# Boxplot by vessel type
ax = axes[1]
data_d = [dwells_df[dwells_df['vtype'] == vt]['dwell_h'].values for vt in vtypes]
bp = ax.boxplot(data_d, patch_artist=True,
                medianprops=dict(color='white', lw=2),
                flierprops=dict(marker='.', markersize=2, alpha=0.3))
for patch, vt in zip(bp['boxes'], vtypes):
    patch.set_facecolor(VTYPE_COLORS.get(vt, '#AAAAAA'))
    patch.set_alpha(0.7)
ax.set_xticklabels(vtypes, rotation=30, ha='right', fontsize=7)
style_ax(ax, 'Dwell Time by Vessel Type', 'Vessel Type', 'Dwell time (hours)')

# Boundary compliance breakdown
ax = axes[2]
below = (dv < 6).sum() / len(dv) * 100
above = (dv > 136).sum() / len(dv) * 100
inside = 100 - below - above
ax.bar(['< 6h\n(below min)', '6–136h\n(valid)', '> 136h\n(above max)'],
       [below, inside, above],
       color=['#E84855', '#3BB273', '#F4A261'], alpha=0.8)
for i, v in enumerate([below, inside, above]):
    ax.text(i, v + 0.3, f'{v:.1f}%', ha='center', fontsize=9, fontweight='bold')
style_ax(ax, 'Dwell Time Boundary Compliance', '', 'Percentage (%)')

plt.tight_layout()
savefig(fig, 'fig04_dwell_distribution.png')

# ══════════════════════════════════════════════════════════
# Fig 05 — Transit time by n_ports group (core validation figure)
# ══════════════════════════════════════════════════════════

print('[Fig 05] Transit time by n_ports group...')
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.patch.set_facecolor('white')
fig.suptitle('Per-Leg Transit Time by n_ports Group\n'
             '(Structural justification for n_ports-stratified transit modelling)',
             fontsize=12, fontweight='bold')

c5, c6, c7 = '#2E86AB', '#E84855', '#3BB273'
groups = [(5, c5), (6, c6), (7, c7)]

# KDE comparison
ax = axes[0]
xs = np.linspace(0, 500, 400)
for n, color in groups:
    sub = legs_df[(legs_df['n_ports'] == n) & (legs_df['transit_h'] <= 500)]['transit_h'].values
    if len(sub) < 10: continue
    kde = gaussian_kde(sub, bw_method=0.25)
    ax.plot(xs, kde(xs), color=color, lw=2,
            label=f'n={n}  med={np.median(sub):.0f}h  (N={len(sub):,})')
    ax.fill_between(xs, kde(xs), alpha=0.2, color=color)

ks56, p56 = ks_2samp(
    legs_df[legs_df['n_ports']==5]['transit_h'].values,
    legs_df[legs_df['n_ports']==6]['transit_h'].values)
ks57, p57 = ks_2samp(
    legs_df[legs_df['n_ports']==5]['transit_h'].values,
    legs_df[legs_df['n_ports']==7]['transit_h'].values)
ax.text(0.97, 0.97,
        f'K-S (n5 vs n6): D={ks56:.3f}, p={p56:.4f}\n'
        f'K-S (n5 vs n7): D={ks57:.3f}, p={p57:.4f}',
        transform=ax.transAxes, fontsize=7.5, va='top', ha='right',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85))
style_ax(ax, 'KDE by n_ports (≤500h)', 'Transit time (hours)', 'Density')
ax.legend(fontsize=8)

# Boxplot
ax = axes[1]
data_np = [legs_df[legs_df['n_ports']==n]['transit_h'].values for n, _ in groups]
bp = ax.boxplot(data_np, patch_artist=True,
                medianprops=dict(color='white', lw=2),
                flierprops=dict(marker='.', markersize=2, alpha=0.3))
for patch, (_, color) in zip(bp['boxes'], groups):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax.set_xticklabels([f'n={n}' for n, _ in groups], fontsize=9)
style_ax(ax, 'Boxplot by n_ports', 'n_ports', 'Transit time (hours)')

# Median and mean comparison bar chart
ax = axes[2]
x = np.arange(len(groups))
medians = [np.median(legs_df[legs_df['n_ports']==n]['transit_h'].values) for n, _ in groups]
means   = [np.mean(legs_df[legs_df['n_ports']==n]['transit_h'].values) for n, _ in groups]
bars1 = ax.bar(x - 0.2, medians, 0.35, label='Median',
               color=[c for _, c in groups], alpha=0.8)
bars2 = ax.bar(x + 0.2, means, 0.35, label='Mean',
               color=[c for _, c in groups], alpha=0.45, hatch='//')
ax.set_xticks(x)
ax.set_xticklabels([f'n={n}' for n, _ in groups])
ax.legend(fontsize=8)
for bar, v in zip(bars1, medians):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+2, f'{v:.0f}h',
            ha='center', fontsize=7.5, fontweight='bold')
style_ax(ax, 'Mean & Median by n_ports', 'n_ports', 'Transit time (hours)')

plt.tight_layout()
savefig(fig, 'fig05_transit_by_nports.png')

# ══════════════════════════════════════════════════════════
# Fig 06 — Transit time by vessel type
# ══════════════════════════════════════════════════════════

print('[Fig 06] Transit time by vessel type...')
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
fig.patch.set_facecolor('white')
fig.suptitle('Per-Leg Transit Time by Vessel Type', fontsize=13, fontweight='bold')
axes_flat = axes.flatten()
xs = np.linspace(0, 500, 400)

for idx, vt in enumerate(vtypes):
    ax = axes_flat[idx]
    for n, color in [(5, '#2E86AB'), (6, '#E84855'), (7, '#3BB273')]:
        sub = legs_df[(legs_df['vtype']==vt) &
                      (legs_df['n_ports']==n) &
                      (legs_df['transit_h']<=500)]['transit_h'].values
        if len(sub) < 5: continue
        try:
            kde = gaussian_kde(sub, bw_method=0.3)
            ax.plot(xs, kde(xs), color=color, lw=1.8, label=f'n={n} (N={len(sub)})')
            ax.fill_between(xs, kde(xs), alpha=0.18, color=color)
        except Exception:
            pass
    n_total = len(legs_df[legs_df['vtype']==vt])
    style_ax(ax, f'{vt}\n({n_total} legs)', 'Transit (h)', 'Density', fontsize=8)
    ax.legend(fontsize=6.5, framealpha=0.8)

for j in range(len(vtypes), len(axes_flat)):
    axes_flat[j].set_visible(False)

plt.tight_layout()
savefig(fig, 'fig06_transit_by_vtype.png')

# ══════════════════════════════════════════════════════════
# Fig 07 — Port visit frequency
# ══════════════════════════════════════════════════════════

print('[Fig 07] Port visit frequency...')
port_counts = {}
for _, row in df.iterrows():
    for p in row['ports'][:-1]:  # exclude NZ destination port
        port_counts[p] = port_counts.get(p, 0) + 1

pc_series = pd.Series(port_counts).sort_values(ascending=False)
top30 = pc_series.head(30)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.patch.set_facecolor('white')
fig.suptitle('Port Visit Frequency (Excluding NZ Destination)', fontsize=13, fontweight='bold')

# Top 30 bar chart
ax = axes[0]
colors_bar = ['#E84855' if i < 5 else '#2E86AB' if i < 15 else '#AAAAAA'
               for i in range(len(top30))]
ax.barh(range(len(top30)), top30.values, color=colors_bar, alpha=0.8)
ax.set_yticks(range(len(top30)))
ax.set_yticklabels(top30.index, fontsize=7.5)
ax.invert_yaxis()
style_ax(ax, 'Top 30 Most Visited Ports', 'Visit count', '')

# Frequency distribution (long-tail analysis)
ax = axes[1]
ax.hist(pc_series.values, bins=50, color='#2E86AB', alpha=0.7)
ax.axvline(pc_series.median(), color='#E84855', lw=1.5, ls='--',
           label=f'Median={pc_series.median():.0f}')
ax.set_yscale('log')
style_ax(ax, f'Port Visit Frequency Distribution\n(total {len(pc_series)} unique ports)',
         'Visit count', 'Number of ports (log scale)')
ax.legend(fontsize=8)

plt.tight_layout()
savefig(fig, 'fig07_port_network.png')

# ══════════════════════════════════════════════════════════
# Fig 08 — New Zealand destination ports
# ══════════════════════════════════════════════════════════

print('[Fig 08] New Zealand destination ports...')
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor('white')
fig.suptitle('New Zealand Destination Port Distribution', fontsize=13, fontweight='bold')

nz_counts = df['nz_dest_port'].value_counts()
colors_nz = PALETTE[:len(nz_counts)]

ax = axes[0]
wedges, texts, autotexts = ax.pie(
    nz_counts.values, labels=nz_counts.index,
    autopct='%1.1f%%', colors=colors_nz,
    startangle=90, pctdistance=0.75,
    textprops={'fontsize': 8}
)
style_ax(ax, 'NZ Destination Port Share')

ax = axes[1]
for i, (vt, color) in enumerate(VTYPE_COLORS.items()):
    sub = df[df['nbic_type_group'] == vt]
    if len(sub) == 0: continue
    nc = sub['nz_dest_port'].value_counts(normalize=True)
    bottom = 0
    for j, (port, share) in enumerate(nc.items()):
        ax.bar(vt, share, bottom=bottom,
               color=PALETTE[j % len(PALETTE)], alpha=0.8,
               label=port if i == 0 else '')
        bottom += share
ax.set_xticklabels(ax.get_xticklabels(), rotation=25, ha='right', fontsize=8)
style_ax(ax, 'NZ Port Share by Vessel Type', 'Vessel Type', 'Share')
handles = [plt.Rectangle((0,0),1,1, color=PALETTE[j % len(PALETTE)])
           for j in range(len(nz_counts))]
ax.legend(handles, nz_counts.index, fontsize=7, loc='upper right')

plt.tight_layout()
savefig(fig, 'fig08_nz_destination.png')

# ══════════════════════════════════════════════════════════
# Fig 09 — Origin port geographic distribution
# ══════════════════════════════════════════════════════════

print('[Fig 09] Origin port geographic distribution...')
origin_lats, origin_lons = [], []
for _, row in df.iterrows():
    if row['latitudes'] and row['longitudes']:
        origin_lats.append(row['latitudes'][0])
        origin_lons.append(row['longitudes'][0])

fig, axes = plt.subplots(1, 2, figsize=(15, 5))
fig.patch.set_facecolor('white')
fig.suptitle('Origin Port Geographic Distribution', fontsize=13, fontweight='bold')

# Scatter plot coloured by vessel type
ax = axes[0]
for vt in vtypes:
    sub = df[df['nbic_type_group'] == vt]
    lats_v = [r['latitudes'][0] for _, r in sub.iterrows() if r['latitudes']]
    lons_v = [r['longitudes'][0] for _, r in sub.iterrows() if r['longitudes']]
    ax.scatter(lons_v, lats_v, c=VTYPE_COLORS.get(vt, '#AAAAAA'),
               alpha=0.4, s=8, label=vt)
ax.axhline(0, color='#888888', lw=0.5, ls='--', alpha=0.5)
ax.set_xlabel('Longitude', fontsize=9)
ax.set_ylabel('Latitude', fontsize=9)
style_ax(ax, 'Origin Ports by Vessel Type', 'Longitude', 'Latitude')
ax.legend(fontsize=6.5, loc='lower left', markerscale=2)

# 2D density heatmap
ax = axes[1]
lats_arr = np.array(origin_lats)
lons_arr = np.array(origin_lons)
h, xedges, yedges = np.histogram2d(lons_arr, lats_arr, bins=40)
ax.imshow(h.T, origin='lower',
          extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
          aspect='auto', cmap='Blues', interpolation='gaussian')
ax.axhline(0, color='white', lw=0.5, ls='--', alpha=0.5)
style_ax(ax, 'Origin Port Density Heatmap', 'Longitude', 'Latitude')

plt.tight_layout()
savefig(fig, 'fig09_origin_geography.png')

# ══════════════════════════════════════════════════════════
# Fig 10 — Ecoprovince coverage
# ══════════════════════════════════════════════════════════

print('[Fig 10] Ecoprovince coverage...')
eco_counts = {}
for _, row in df.iterrows():
    for eco in row['ecoprovinces'][:-1]:  # exclude NZ destination port
        if eco and str(eco) not in ('', 'nan'):
            eco_counts[str(eco)] = eco_counts.get(str(eco), 0) + 1

ec = pd.Series(eco_counts).sort_values(ascending=False)
top20_eco = ec.head(20)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.patch.set_facecolor('white')
fig.suptitle('Ecoprovince Coverage (Source Ports)', fontsize=13, fontweight='bold')

ax = axes[0]
ax.barh(range(len(top20_eco)), top20_eco.values,
        color='#2E86AB', alpha=0.75)
ax.set_yticks(range(len(top20_eco)))
ax.set_yticklabels(top20_eco.index, fontsize=7.5)
ax.invert_yaxis()
style_ax(ax, f'Top 20 Ecoprovinces\n({len(ec)} unique provinces total)',
         'Visit count', '')

ax = axes[1]
ax.bar(range(len(ec)), ec.values, color='#2E86AB', alpha=0.7, width=1.0)
ax.set_xlabel('Ecoprovince rank', fontsize=9)
ax.set_ylabel('Visit count', fontsize=9)
ax.set_yscale('log')
style_ax(ax, 'All Ecoprovinces — Long Tail',
         'Ecoprovince rank (sorted)', 'Visit count (log scale)')

plt.tight_layout()
savefig(fig, 'fig10_ecoprovince.png')

# ══════════════════════════════════════════════════════════
# Fig 11 — Temporal analysis
# ══════════════════════════════════════════════════════════

if 'start_month' in df.columns and df['start_month'].notna().sum() > 100:
    print('[Fig 11] Temporal analysis...')
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor('white')
    fig.suptitle('Temporal Analysis of Voyages', fontsize=13, fontweight='bold')

    month_names = ['Jan','Feb','Mar','Apr','May','Jun',
                   'Jul','Aug','Sep','Oct','Nov','Dec']

    # Monthly distribution
    ax = axes[0]
    mc = df['start_month'].value_counts().sort_index()
    ax.bar(mc.index, mc.values, color='#2E86AB', alpha=0.8)
    ax.set_xticks(range(1,13))
    ax.set_xticklabels(month_names, fontsize=7.5, rotation=30)
    style_ax(ax, 'Voyage Count by Start Month', 'Month', 'Count')

    # Annual distribution
    ax = axes[1]
    yc = df['start_year'].value_counts().sort_index().dropna()
    ax.bar(yc.index.astype(int).astype(str), yc.values,
           color='#E84855', alpha=0.8)
    style_ax(ax, 'Voyage Count by Year', 'Year', 'Count')

    # Month × vessel type heatmap
    ax = axes[2]
    pivot = df.groupby(['start_month', 'nbic_type_group']).size().unstack(fill_value=0)
    im = ax.imshow(pivot.T.values, aspect='auto', cmap='Blues',
                   interpolation='nearest')
    ax.set_xticks(range(len(pivot.index)))
    ax.set_xticklabels(month_names, fontsize=7, rotation=30)
    ax.set_yticks(range(len(pivot.columns)))
    ax.set_yticklabels(pivot.columns, fontsize=7.5)
    plt.colorbar(im, ax=ax, shrink=0.8)
    style_ax(ax, 'Voyage Count Heatmap\n(Month × Vessel Type)', 'Month', '')

    plt.tight_layout()
    savefig(fig, 'fig11_temporal.png')
else:
    print('[Fig 11] Insufficient temporal data — skipped')

# ══════════════════════════════════════════════════════════
# Fig 12 — DWT by vessel type (detailed)
# ══════════════════════════════════════════════════════════

print('[Fig 12] DWT by vessel type (detailed)...')
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
fig.patch.set_facecolor('white')
fig.suptitle('DWT Distribution by Vessel Type', fontsize=13, fontweight='bold')
axes_flat = axes.flatten()

for idx, vt in enumerate(vtypes):
    ax = axes_flat[idx]
    sub = df[df['nbic_type_group'] == vt]['DWT'].values / 1e3
    if len(sub) < 3:
        ax.set_visible(False)
        continue
    xs = np.linspace(sub.min(), sub.max(), 300)
    try:
        kde = gaussian_kde(sub, bw_method=0.3)
        ax.plot(xs, kde(xs), color=VTYPE_COLORS.get(vt, '#2E86AB'), lw=2)
        ax.fill_between(xs, kde(xs), alpha=0.35,
                        color=VTYPE_COLORS.get(vt, '#2E86AB'))
    except Exception:
        pass
    ax.axvline(np.median(sub), color='#333333', lw=1.5, ls='--',
               label=f'Med={np.median(sub):.0f}k')
    style_ax(ax, f'{vt}\n(n={len(sub):,})',
             'DWT (×10³ tonnes)', 'Density', fontsize=8)
    ax.legend(fontsize=7)

for j in range(len(vtypes), len(axes_flat)):
    axes_flat[j].set_visible(False)

plt.tight_layout()
savefig(fig, 'fig12_dwt_by_vtype.png')

# ══════════════════════════════════════════════════════════
# Fig 13 — Correlation matrix of key variables
# ══════════════════════════════════════════════════════════

print('[Fig 13] Correlation matrix...')
corr_cols = {
    'n_ports':        'n_ports',
    'total_span_days': 'Span (days)',
    'DWT':            'DWT',
    'delta_t_hours':  'Δt (hours)',
}
corr_df = df[list(corr_cols.keys())].copy()
corr_df.columns = list(corr_cols.values())

# Add per-voyage average transit and dwell
corr_df2 = corr_df.copy()
corr_df2['avg_transit_h'] = df.apply(
    lambda r: np.mean([t for t in r['transits_hours'][1:] if t > 0]) if len(r['transits_hours']) > 1 else np.nan,
    axis=1
)
corr_df2['avg_dwell_h'] = df.apply(
    lambda r: np.mean([d for d in r['dwells_hours'][1:-1] if d > 0]) if len(r['dwells_hours']) > 2 else np.nan,
    axis=1
)
corr_df2.columns = list(corr_cols.values()) + ['Avg Transit (h)', 'Avg Dwell (h)']
corr_matrix = corr_df2.corr()

fig, ax = plt.subplots(figsize=(8, 7))
fig.patch.set_facecolor('white')
cmap = LinearSegmentedColormap.from_list('rw', ['#E84855', 'white', '#2E86AB'])
im = ax.imshow(corr_matrix.values, vmin=-1, vmax=1, cmap=cmap, aspect='auto')
plt.colorbar(im, ax=ax, shrink=0.85)
n = len(corr_matrix)
ax.set_xticks(range(n))
ax.set_yticks(range(n))
ax.set_xticklabels(corr_matrix.columns, rotation=30, ha='right', fontsize=9)
ax.set_yticklabels(corr_matrix.columns, fontsize=9)
for i in range(n):
    for j in range(n):
        val = corr_matrix.values[i, j]
        ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                fontsize=8, color='black' if abs(val) < 0.7 else 'white')
style_ax(ax, 'Correlation Matrix of Key Voyage Variables')
plt.tight_layout()
savefig(fig, 'fig13_correlation.png')

# ══════════════════════════════════════════════════════════
# Fig 14 — Delta-t distribution
# ══════════════════════════════════════════════════════════

print('[Fig 14] Δt distribution...')
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.patch.set_facecolor('white')
fig.suptitle('Δt (Cumulative Transit+Dwell from Origin to NZ)', fontsize=13, fontweight='bold')

dt = df['delta_t_hours'].values
dt_clip = dt[dt <= 2000]

# KDE
ax = axes[0]
xs = np.linspace(0, 2000, 400)
kde = gaussian_kde(dt_clip, bw_method=0.25)
ax.plot(xs, kde(xs), color='#9B5DE5', lw=2)
ax.fill_between(xs, kde(xs), alpha=0.3, color='#9B5DE5')
ax.axvline(np.median(dt_clip), color='#E84855', lw=1.5, ls='--',
           label=f'Median={np.median(dt_clip):.0f}h')
style_ax(ax, 'Δt KDE (≤2000h)', 'Δt (hours)', 'Density')
ax.legend(fontsize=8)

# Δt grouped by n_ports
ax = axes[1]
for n, color in [(5, '#2E86AB'), (6, '#E84855'), (7, '#3BB273')]:
    sub = df[df['n_ports'] == n]['delta_t_hours'].values
    sub_c = sub[sub <= 2000]
    if len(sub_c) < 5: continue
    kde = gaussian_kde(sub_c, bw_method=0.3)
    ax.plot(xs, kde(xs), color=color, lw=2, label=f'n={n} med={np.median(sub_c):.0f}h')
    ax.fill_between(xs, kde(xs), alpha=0.2, color=color)
style_ax(ax, 'Δt by n_ports', 'Δt (hours)', 'Density')
ax.legend(fontsize=8)

# Δt vs total_span_days scatter
ax = axes[2]
ax.scatter(df['delta_t_hours']/24, df['total_span_days'],
           c=[VTYPE_COLORS.get(v, '#AAAAAA') for v in df['nbic_type_group']],
           alpha=0.3, s=8)
ax.plot([0, 60], [0, 60], color='#333333', lw=1, ls='--', alpha=0.5,
        label='1:1 line')
style_ax(ax, 'Δt vs Total Span', 'Δt (days)', 'Total span (days)')
ax.legend(fontsize=8)

plt.tight_layout()
savefig(fig, 'fig14_delta_t.png')

# ══════════════════════════════════════════════════════════
# Done
# ══════════════════════════════════════════════════════════

print(f'\nAll figures saved to: {OUT_DIR}')
print(f'14 figures generated.')