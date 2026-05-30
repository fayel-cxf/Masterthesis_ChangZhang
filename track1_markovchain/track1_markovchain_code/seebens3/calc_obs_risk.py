"""
calc_obs_risk.py
================
Observed voyage risk calculation — based on risk_calculation_mc.py
Outputs summary statistics + figures to data/seebens/obsrisk/

Run: python code/seebens/calc_obs_risk.py
"""

import ast
import pickle
import re
import warnings
from math import radians, sin, cos, sqrt, atan2
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════
# 0. Configuration
# ══════════════════════════════════════════════════════════

BASE_DIR = Path('D:/2025ChangZhang')
OBS_PATH = BASE_DIR / 'data/clustering/cluster.m1.labels.k5.csv'
ESR_PATH = BASE_DIR / 'data/seebens/ecoprov_envdist_scaled.csv'
REG_PATH = BASE_DIR / 'data/seebens/log_regression_results.csv'
SEA_PATH = BASE_DIR / 'code/markov/sea_distance_matrix.pkl'
OUT_DIR  = BASE_DIR / 'data/seebens/obsrisk'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Risk parameters — identical to risk_calculation_mc.py
BETA        = 8.0
GAMMA       = 1000.0
MU          = 0.02
LAMBDA_BW   = 0.002
RHO         = 1.0
Z           = 0.95
VR_COEFF    = 0.3
ALPHA       = 1.5e-4
ESR_INTRA   = 0.99      # diagonal override (same province)
GC_CORRECTION = 1.3
P_SIG       = 0.05

NZ_PORT_PROV = {
    'AUCKLAND': 53, 'TAURANGA': 53, 'GISBORNE': 53, 'WHANGAREI': 53,
    'NAPIER': 54, 'NELSON': 54, 'NEW PLYMOUTH': 54, 'PORT CHALMERS': 54,
    'LYTTELTON': 54, 'TIMARU': 54, 'WELLINGTON': 54, 'BLUFF': 54,
}

ECOPROV_NAME_TO_ID = {
    'Arctic': 1, 'Northern European Seas': 2, 'Lusitanian': 3,
    'Mediterranean Sea': 4, 'Cold Temperate Northwest Atlantic': 5,
    'Warm Temperate Northwest Atlantic': 6, 'Black Sea': 7,
    'Cold Temperate Northwest Pacific': 8, 'Warm Temperate Northwest Pacific': 9,
    'Cold Temperate Northeast Pacific': 10, 'Warm Temperate Northeast Pacific': 11,
    'Tropical Northwestern Atlantic': 12, 'Tropical Southwestern Atlantic': 14,
    'West African Transition': 16, 'Gulf of Guinea': 17,
    'Red Sea and Gulf of Aden': 18, 'Somali/Arabian': 19,
    'Western Indian Ocean': 20, 'West and South Indian Shelf': 21,
    'Bay of Bengal': 23, 'Andaman': 24, 'South China Sea': 25,
    'Sunda Shelf': 26, 'Java Transitional': 27, 'South Kuroshio': 28,
    'Tropical Northwestern Pacific': 29, 'Western Coral Triangle': 30,
    'Eastern Coral Triangle': 31, 'Sahul Shelf': 32,
    'Northeast Australian Shelf': 33, 'Northwest Australian Shelf': 34,
    'Tropical Southwestern Pacific': 35, 'Lord Howe and Norfolk Islands': 36,
    'Hawaii': 37, 'Marshall, Gilbert and Ellis Islands': 38,
    'Central Polynesia': 39, 'Southeast Polynesia': 40,
    'Tropical East Pacific': 43, 'Warm Temperate Southeastern Pacific': 45,
    'Warm Temperate Southwestern Atlantic': 47, 'Magellanic': 48,
    'Benguela': 50, 'Agulhas': 51, 'Northern New Zealand': 53,
    'Southern New Zealand': 54, 'East Central Australian Shelf': 55,
    'Southeast Australian Shelf': 56, 'Southwest Australian Shelf': 57,
    'West Central Australian Shelf': 58,
}

# ══════════════════════════════════════════════════════════
# 1. Utilities
# ══════════════════════════════════════════════════════════

def safe_eval(val):
    if pd.isna(val):
        return []
    s = re.sub(r'np\.\w+\(([^)]+)\)', r'\1', str(val))
    try:
        return ast.literal_eval(s)
    except Exception:
        return []

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1r, lat2r = radians(lat1), radians(lat2)
    dlat = lat2r - lat1r
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(lat1r)*cos(lat2r)*sin(dlon/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))

# ══════════════════════════════════════════════════════════
# 2. Load data
# ══════════════════════════════════════════════════════════

print('=' * 65)
print('Observed Voyage Risk Calculation')
print('=' * 65)

print('\n[Load] ESR matrix...')
esr_df = pd.read_csv(ESR_PATH, index_col=0)
esr_df.index   = esr_df.index.astype(int)
esr_df.columns = esr_df.columns.astype(int)
print(f'       {esr_df.shape[0]}×{esr_df.shape[1]} province pairs')

print('\n[Load] W_r regression coefficients...')
reg = pd.read_csv(REG_PATH)
reg_model = {}
sig_mask = reg['p_value'] < P_SIG
fallback_intercept = float(reg[sig_mask]['intercept'].mean())
fallback_slope     = float(reg[sig_mask]['slope'].mean())
for _, r in reg.iterrows():
    if r['p_value'] < P_SIG:
        reg_model[r['group']] = (float(r['intercept']), float(r['slope']))
        status = 'significant'
    else:
        reg_model[r['group']] = (fallback_intercept, fallback_slope)
        status = f'NON-SIGNIFICANT (p={r["p_value"]:.3f}) → fallback'
    print(f'       {r["group"]:15s}  intercept={r["intercept"]:7.4f}  '
          f'slope={r["slope"]:7.4f}  {status}')
reg_model['_default'] = (fallback_intercept, fallback_slope)

print('\n[Load] Sea distance matrix...')
try:
    with open(SEA_PATH, 'rb') as f:
        sea_raw = pickle.load(f)
    sea_nm = sea_raw['nm']
    print(f'       {len(sea_nm):,} port pairs (nm)')
except Exception as e:
    sea_nm = {}
    print(f'       WARNING: {e} — using GC fallback')

print('\n[Load] Observed voyages...')
LIST_COLS = ['ports', 'latitudes', 'longitudes',
             'ecoprovinces', 'transits_hours', 'dwells_hours']
obs = pd.read_csv(OBS_PATH)
for col in LIST_COLS:
    if col in obs.columns:
        obs[col] = obs[col].apply(safe_eval)
if 'nbic_type_group' in obs.columns and 'vessel_type' not in obs.columns:
    obs = obs.rename(columns={'nbic_type_group': 'vessel_type'})
print(f'       {len(obs)} voyages')

port_coords = {}
for row in obs.itertuples(index=False):
    for p, la, lo in zip(row.ports, row.latitudes, row.longitudes):
        if p not in port_coords:
            try:
                port_coords[p] = (float(la), float(lo))
            except (TypeError, ValueError):
                pass

# ══════════════════════════════════════════════════════════
# 3. Risk functions (identical to risk_calculation_mc.py)
# ══════════════════════════════════════════════════════════

def get_dist_km(port_a, port_b):
    for key in [(port_a, port_b), (port_b, port_a)]:
        if key in sea_nm:
            return float(sea_nm[key]) / 0.5399
    ca = port_coords.get(port_a)
    cb = port_coords.get(port_b)
    if ca is None or cb is None:
        return None
    return haversine_km(ca[0], ca[1], cb[0], cb[1]) * GC_CORRECTION

def p_alien(src_port, nz_port):
    ca = port_coords.get(src_port)
    cb = port_coords.get(nz_port)
    if ca is None or cb is None:
        return 0.5
    d = haversine_km(ca[0], ca[1], cb[0], cb[1])
    return float(1.0 / (1.0 + np.exp(-BETA * (d - GAMMA) / 1000.0)))

def predict_wr(vessel_type, dwt):
    intercept, slope = reg_model.get(vessel_type, reg_model['_default'])
    return float(np.exp(intercept + slope * np.log(max(dwt, 1.0))))

def p_intro(delta_t_hours, delta_r, dwt, vessel_type):
    W_r = predict_wr(vessel_type, dwt)
    V_r = VR_COEFF * max(dwt, 1.0)
    if V_r <= 0 or W_r <= 0:
        return 0.0
    exchange_ratio = min(Z * W_r / V_r, 0.9999)
    B_r  = Z * W_r * ((1.0 - exchange_ratio) ** delta_r)
    prob = RHO * (1.0 - np.exp(-LAMBDA_BW * B_r)) * np.exp(-MU * delta_t_hours / 24.0)
    return float(np.clip(prob, 0.0, 1.0))

def p_estab(src_eco, nz_eco):
    src_id = ECOPROV_NAME_TO_ID.get(src_eco)
    nz_id  = ECOPROV_NAME_TO_ID.get(nz_eco)
    if src_id is None or nz_id is None:
        return 0.0
    if src_id == nz_id:
        esr_val = ESR_INTRA
    elif src_id in esr_df.index and nz_id in esr_df.columns:
        esr_val = float(esr_df.loc[src_id, nz_id])
    else:
        return 0.0
    return float(ALPHA * esr_val)

def compute_voyage_risk(ports, ecoprovinces, transits, dwells,
                        n_ports, dwt, vessel_type):
    n       = n_ports - 1
    nz_port = ports[n]
    nz_eco  = ecoprovinces[n] if n < len(ecoprovinces) else ''
    pa_list, pi_list, pe_list, pr_list = [], [], [], []

    for i in range(n):
        src_port = ports[i]
        src_eco  = ecoprovinces[i] if i < len(ecoprovinces) else ''
        pa = p_alien(src_port, nz_port)
        delta_t_h = sum(transits[i+1: n+1]) + sum(dwells[i+1: n])
        delta_r   = n - i - 1
        pi = p_intro(delta_t_h, delta_r, dwt, vessel_type)
        pe = p_estab(src_eco, nz_eco)
        pr = pa * pi * pe
        pa_list.append(round(pa, 6))
        pi_list.append(round(pi, 8))
        pe_list.append(round(pe, 8))
        pr_list.append(round(pr, 10))

    p_inv = float(1.0 - np.prod([1.0 - p for p in pr_list]))
    return {
        'p_alien_per_port': pa_list,
        'p_intro_per_port': pi_list,
        'p_estab_per_port': pe_list,
        'pr_inv_per_port':  pr_list,
        'p_invasion':       round(p_inv, 10),
    }

# ══════════════════════════════════════════════════════════
# 4. Compute risk for all observed voyages
# ══════════════════════════════════════════════════════════

print('\n[Risk] Computing observed voyage risk...')
results, hop_records = [], []

for row in obs.itertuples(index=False):
    n_ports = int(row.n_ports)
    if n_ports < 2 or len(row.ports) != n_ports:
        continue
    risk = compute_voyage_risk(
        ports        = row.ports,
        ecoprovinces = row.ecoprovinces,
        transits     = row.transits_hours,
        dwells       = row.dwells_hours,
        n_ports      = n_ports,
        dwt          = float(row.DWT),
        vessel_type  = str(row.vessel_type),
    )
    results.append({
        'voyage_id':    row.voyage_id,
        'vessel_type':  row.vessel_type,
        'cluster':      row.cluster,
        'dwt':          row.DWT,
        'n_ports':      n_ports,
        'nz_dest_port': row.nz_dest_port,
        'total_span_days': row.total_span_days,
        'p_invasion':   risk['p_invasion'],
        'pa_mean': float(np.mean(risk['p_alien_per_port'])),
        'pi_mean': float(np.mean(risk['p_intro_per_port'])),
        'pe_mean': float(np.mean(risk['p_estab_per_port'])),
    })
    # Hop records for dilution decay
    n = n_ports - 1
    for i, pr in enumerate(risk['pr_inv_per_port']):
        hop_records.append({
            'hops_to_nz': n - i,
            'pr_inv': pr,
            'pa': risk['p_alien_per_port'][i],
            'pi': risk['p_intro_per_port'][i],
            'pe': risk['p_estab_per_port'][i],
        })

df_res  = pd.DataFrame(results)
df_hops = pd.DataFrame(hop_records)
pj      = df_res['p_invasion'].values
print(f'       Processed: {len(df_res)} voyages')

# ══════════════════════════════════════════════════════════
# 5. Summary statistics
# ══════════════════════════════════════════════════════════

print('\n' + '=' * 65)
print('SUMMARY STATISTICS')
print('=' * 65)
stats = [
    ('n',      len(pj)),
    ('mean',   np.mean(pj)),
    ('median', np.median(pj)),
    ('std',    np.std(pj)),
    ('p5',     np.percentile(pj, 5)),
    ('p90',    np.percentile(pj, 90)),
    ('p95',    np.percentile(pj, 95)),
    ('max',    np.max(pj)),
    ('min',    np.min(pj)),
]
for k, v in stats:
    if k == 'n':
        print(f'  {k:<10}: {int(v)}')
    else:
        print(f'  {k:<10}: {v:.6e}   (x10^-4: {v*1e4:.4f})')

# ══════════════════════════════════════════════════════════
# 6. Save CSVs
# ══════════════════════════════════════════════════════════

df_res.to_csv(OUT_DIR / 'obs_risk_results.csv', index=False)
pd.DataFrame([dict(stats)]).to_csv(OUT_DIR / 'obs_risk_summary.csv', index=False)

df_port = (df_res.groupby('nz_dest_port')['p_invasion']
           .agg(n='count', mean=np.mean, median=np.median,
                std=np.std, p90=lambda x: np.percentile(x, 90),
                p95=lambda x: np.percentile(x, 95))
           .sort_values('mean', ascending=False).reset_index())
df_port.to_csv(OUT_DIR / 'obs_risk_by_port.csv', index=False)

df_type = (df_res.groupby('vessel_type')['p_invasion']
           .agg(n='count', mean=np.mean, median=np.median, std=np.std)
           .sort_values('mean', ascending=False).reset_index())
df_type.to_csv(OUT_DIR / 'obs_risk_by_type.csv', index=False)

df_hop = (df_hops.groupby('hops_to_nz')['pr_inv']
          .agg(n='count', mean=np.mean, median=np.median, std=np.std)
          .sort_index().reset_index())
df_hop.to_csv(OUT_DIR / 'obs_risk_by_hop.csv', index=False)

print(f'\n[Save] CSVs written to {OUT_DIR}')

# ══════════════════════════════════════════════════════════
# 7. Figures
# ══════════════════════════════════════════════════════════

BLUE = '#2C6E9E'
FIG_DPI = 150

def save_fig(name):
    plt.savefig(OUT_DIR / name, dpi=FIG_DPI, bbox_inches='tight')
    plt.close()
    print(f'[Save] {name}')

# Fig 1: Distribution — histogram + CDF + stats
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

ax = axes[0]
ax.hist(pj * 1e4, bins=40, color=BLUE, alpha=0.7, edgecolor='white')
ax.axvline(np.mean(pj)*1e4,   color='red',    lw=1.5, ls='--',
           label=f'Mean={np.mean(pj)*1e4:.4f}')
ax.axvline(np.median(pj)*1e4, color='orange', lw=1.5, ls=':',
           label=f'Median={np.median(pj)*1e4:.4f}')
ax.set_xlabel('$P_j$ (×10$^{-4}$)')
ax.set_ylabel('Count')
ax.set_title('Frequency Histogram')
ax.legend(fontsize=8)

ax = axes[1]
sorted_pj = np.sort(pj) * 1e4
cdf = np.arange(1, len(sorted_pj)+1) / len(sorted_pj)
ax.plot(sorted_pj, cdf, color=BLUE, lw=2)
ax.axvline(np.percentile(pj, 90)*1e4, color='orange', lw=1.2, ls='--',
           label=f'P90={np.percentile(pj,90)*1e4:.4f}')
ax.axvline(np.percentile(pj, 95)*1e4, color='red',    lw=1.2, ls='--',
           label=f'P95={np.percentile(pj,95)*1e4:.4f}')
ax.set_xlabel('$P_j$ (×10$^{-4}$)')
ax.set_ylabel('Cumulative probability')
ax.set_title('Empirical CDF')
ax.legend(fontsize=8)

ax = axes[2]
ax.axis('off')
lines = [f'{k:<10}: {int(v) if k=="n" else f"{v:.6e}  (x10^-4: {v*1e4:.4f})"}'
         for k, v in stats]
ax.text(0.05, 0.95, '\n'.join(lines), transform=ax.transAxes,
        fontsize=8.5, va='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
ax.set_title('Summary Statistics')

fig.suptitle('Observed Voyage Risk Distribution ($P_j$, n=2,432)', fontsize=12)
plt.tight_layout()
save_fig('fig_distribution.png')

# Fig 2: Per-port ranking
fig, ax = plt.subplots(figsize=(10, 5))
colors = [BLUE if i < 3 else '#888888' for i in range(len(df_port))]
bars = ax.bar(df_port['nz_dest_port'], df_port['mean']*1e4,
              color=colors, alpha=0.85, edgecolor='white')
ax.errorbar(df_port['nz_dest_port'], df_port['mean']*1e4,
            yerr=df_port['std']*1e4, fmt='none', color='black', capsize=4)
for bar, val in zip(bars, df_port['mean']):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
            f'{val*1e4:.4f}', ha='center', va='bottom', fontsize=8)
ax.axhline(np.mean(pj)*1e4, color='red', ls='--', lw=1.2,
           label=f'Overall mean={np.mean(pj)*1e4:.4f}')
ax.set_xlabel('NZ Destination Port')
ax.set_ylabel('Mean $P_j$ (×10$^{-4}$)')
ax.set_title('Per-Port Risk Ranking (Observed Voyages)')
ax.legend(fontsize=9)
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
save_fig('fig_port_ranking.png')

# Fig 3: Per-vessel-type boxplot
fig, ax = plt.subplots(figsize=(10, 5))
type_order = df_type['vessel_type'].tolist()
data_by_type = [df_res[df_res['vessel_type']==t]['p_invasion'].values*1e4
                for t in type_order]
bp = ax.boxplot(data_by_type, labels=type_order, patch_artist=True,
                medianprops=dict(color='red', linewidth=2))
for patch in bp['boxes']:
    patch.set_facecolor(BLUE)
    patch.set_alpha(0.6)
ax.axhline(np.mean(pj)*1e4, color='red', ls='--', lw=1.2,
           label=f'Overall mean={np.mean(pj)*1e4:.4f}')
for i, t in enumerate(type_order):
    n_t = len(df_res[df_res['vessel_type']==t])
    ax.text(i+1, ax.get_ylim()[0], f'n={n_t}', ha='center', va='bottom',
            fontsize=8, color='grey')
ax.set_xlabel('Vessel Type')
ax.set_ylabel('$P_j$ (×10$^{-4}$)')
ax.set_title('Per-Vessel-Type Risk (Observed Voyages)')
ax.legend(fontsize=9)
plt.tight_layout()
save_fig('fig_vessel_type.png')

# Fig 4: Dilution decay
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
hops  = df_hop['hops_to_nz'].values
means = df_hop['mean'].values * 1e5
stds  = df_hop['std'].values * 1e5

ax = axes[0]
ax.plot(hops, means, 'o-', color=BLUE, lw=2, ms=7)
ax.fill_between(hops, means-stds, means+stds, color=BLUE, alpha=0.15)
ax.set_xlabel('Hops to New Zealand')
ax.set_ylabel('Mean Pr(Inv) per donor (×10$^{-5}$)')
ax.set_title('Dilution Decay — Absolute')
ax.invert_xaxis(); ax.set_xticks(hops)

ax = axes[1]
norm = means / means[np.argmax(hops)]
ax.plot(hops, norm, 's-', color=BLUE, lw=2, ms=7)
for x, y in zip(hops, norm):
    ax.annotate(f'{y:.2f}', (x, y), textcoords='offset points',
                xytext=(0, 8), ha='center', fontsize=8)
ax.set_xlabel('Hops to New Zealand')
ax.set_ylabel('Normalised mean Pr(Inv)')
ax.set_title('Dilution Decay — Normalised')
ax.invert_xaxis(); ax.set_xticks(hops)

plt.tight_layout()
save_fig('fig_dilution_decay.png')

# Fig 5: Filter component histograms
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
for ax, (col, label, color) in zip(axes, [
    ('pa_mean', '$P_{alien}$',   '#E07B54'),
    ('pi_mean', '$P_{intro}$',   '#5B9BD5'),
    ('pe_mean', '$P_{estab}^*$', '#70AD47'),
]):
    vals = df_res[col].dropna().values
    ax.hist(vals, bins=35, color=color, alpha=0.75, edgecolor='white')
    ax.axvline(np.mean(vals), color='black', ls='--', lw=1.5,
               label=f'Mean={np.mean(vals):.3e}')
    ax.set_xlabel(label); ax.set_ylabel('Count')
    ax.set_title(f'{label} distribution')
    ax.legend(fontsize=8)
fig.suptitle('Filter Component Distributions (Observed)', fontsize=12)
plt.tight_layout()
save_fig('fig_filter_components.png')

print(f'\n✅  All outputs saved to: {OUT_DIR}')