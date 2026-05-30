"""
filter_component_by_hop_track1.py
===================================
Track 1 — Per-hop filter component breakdown (P_alien, P_intro, P_estab)
Mirrors Track 2's Figure 14: hop-wise mean ± 1SE for each filter component,
Observed vs Synthetic side-by-side on three panels.

Prerequisite:
  Run risk_calculation_mc.py first to produce:
    data/seebens/track1_risk_results.csv

Logic:
  For a voyage with n_ports stops (last = NZ destination),
  source port at index i has:
      hops_to_nz = n_ports - 1 - i
  e.g., the port immediately before NZ has hops_to_nz = 1,
        two ports before NZ has hops_to_nz = 2, etc.

Output:
  data/seebens/figures/track1_filter_component_by_hop.png
  data/seebens/figures/track1_filter_component_by_hop.pdf
"""

import ast
import warnings
from io import BytesIO
from pathlib import Path

import matplotlib
matplotlib.use('Agg')                       # avoid WPS file-lock issues
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════
# 0. Paths  (change BASE_DIR only)
# ══════════════════════════════════════════════════════════

BASE_DIR    = Path(__file__).resolve().parents.parent           # D:/2025ChangZhang
RISK_PATH   = BASE_DIR / 'data/seebens/track1_risk_results.csv'
FIG_DIR     = BASE_DIR / 'data/seebens/figures'
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── Plot style parameters ─────────────────────────────────
C_OBS   = '#2166AC'     # observed  — steel blue
C_SYN   = '#D6604D'     # synthetic — brick red
HOP_MIN = 1
HOP_MAX = 6             # hops beyond 6 have very few data points

# ══════════════════════════════════════════════════════════
# 1. Load risk results
# ══════════════════════════════════════════════════════════

print('[Load] Reading risk results...')
df = pd.read_csv(RISK_PATH)
print(f'       {len(df):,} rows  |  sources: {df["source"].value_counts().to_dict()}')


def safe_eval(val):
    """Parse stringified Python list back to a list."""
    if isinstance(val, list):
        return val
    try:
        return ast.literal_eval(str(val))
    except Exception:
        return []


LIST_COLS = ['p_alien_per_port', 'p_intro_per_port', 'p_estab_per_port', 'pr_inv_per_port']
for col in LIST_COLS:
    if col in df.columns:
        df[col] = df[col].apply(safe_eval)

# ══════════════════════════════════════════════════════════
# 2. Explode per-port lists → (voyage_id, hop, pa, pi, pe)
# ══════════════════════════════════════════════════════════

print('[Build] Exploding per-port risk lists...')

records = []
for row in df.itertuples(index=False):
    n = int(row.n_ports)
    pa_list = row.p_alien_per_port
    pi_list = row.p_intro_per_port
    pe_list = row.p_estab_per_port

    # guard: list length must match source-port count (n-1)
    n_src = n - 1
    if (len(pa_list) != n_src or
            len(pi_list) != n_src or
            len(pe_list) != n_src):
        continue

    for i in range(n_src):
        hops = n_src - i        # index 0 → farthest (n_src hops), last → 1 hop
        records.append({
            'source':       row.source,
            'voyage_id':    row.voyage_id,
            'hops_to_nz':   hops,
            'p_alien':      pa_list[i],
            'p_intro':      pi_list[i],
            'p_estab':      pe_list[i],
        })

port_df = pd.DataFrame(records)
print(f'       {len(port_df):,} port-level records exploded')

# Filter to displayable hop range
port_df = port_df[port_df['hops_to_nz'].between(HOP_MIN, HOP_MAX)].copy()
print(f'       {len(port_df):,} records in hop range [{HOP_MIN}, {HOP_MAX}]')

# ══════════════════════════════════════════════════════════
# 3. Aggregate: mean ± 1SE per (source, hop)
# ══════════════════════════════════════════════════════════

def hop_stats(sub: pd.DataFrame, col: str) -> pd.DataFrame:
    """Return DataFrame with columns [hops_to_nz, mean, se] for one filter."""
    rows = []
    for hop, grp in sub.groupby('hops_to_nz'):
        vals = grp[col].dropna().values
        if len(vals) == 0:
            continue
        m  = float(np.mean(vals))
        se = float(stats.sem(vals)) if len(vals) > 1 else 0.0
        rows.append({'hops_to_nz': hop, 'mean': m, 'se': se, 'n': len(vals)})
    return pd.DataFrame(rows).sort_values('hops_to_nz').reset_index(drop=True)


obs = port_df[port_df['source'] == 'observed']
syn = port_df[port_df['source'] == 'synthetic']

stats_obs = {col: hop_stats(obs, col) for col in ['p_alien', 'p_intro', 'p_estab']}
stats_syn = {col: hop_stats(syn, col) for col in ['p_alien', 'p_intro', 'p_estab']}

# Print sample counts
print('\n[Stats] Hop-level counts (observed / synthetic):')
print(f'  {"Hop":>4}  {"Obs N":>8}  {"Syn N":>8}')
for hop in range(HOP_MIN, HOP_MAX + 1):
    n_obs = len(obs[obs['hops_to_nz'] == hop])
    n_syn = len(syn[syn['hops_to_nz'] == hop])
    print(f'  {hop:>4}  {n_obs:>8,}  {n_syn:>8,}')

# ══════════════════════════════════════════════════════════
# 4. Plot — three panels (mirrors Figure 14)
# ══════════════════════════════════════════════════════════

print('\n[Plot] Generating figure...')

fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
fig.patch.set_facecolor('white')

fig.suptitle(
    r'Track 1 Risk Result — Risk Component Breakdown by Hops to NZ'
    r' ($P_\mathrm{alien}$,  $P_\mathrm{intro}$,  $P_\mathrm{estab}$)',
    fontsize=10.5, fontweight='normal', y=1.01
)

# ── panel spec: (col, title_label, ylabel, scale, fmt) ────
panel_specs = [
    ('p_alien',
     r'(a) $P_\mathrm{alien}$',
     r'$P_\mathrm{alien}$',
     1.0,
     '{:.4f}'),
    ('p_intro',
     r'(b) $P_\mathrm{intro}$',
     r'$P_\mathrm{intro}$',
     1.0,
     '{:.2f}'),
    ('p_estab',
     r'(c) $P_\mathrm{estab}$',
     r'$P_\mathrm{estab}$',
     1.0,
     None),        # scientific notation chosen automatically
]

hops = list(range(HOP_MIN, HOP_MAX + 1))


def _get(stats_dict: dict, col: str, hop: int, field: str, scale: float):
    """Safe lookup: return scaled value or NaN."""
    sub = stats_dict[col]
    row = sub[sub['hops_to_nz'] == hop]
    if row.empty:
        return np.nan
    return float(row.iloc[0][field]) * scale


for ax, (col, title, ylabel, scale, fmt) in zip(axes, panel_specs):

    # ── build arrays ──────────────────────────────────────
    obs_mean = np.array([_get(stats_obs, col, h, 'mean', scale) for h in hops])
    obs_se   = np.array([_get(stats_obs, col, h, 'se',   scale) for h in hops])
    syn_mean = np.array([_get(stats_syn, col, h, 'mean', scale) for h in hops])
    syn_se   = np.array([_get(stats_syn, col, h, 'se',   scale) for h in hops])

    # ── error bars (capsize style matches Track 2 figure) ─
    ax.errorbar(hops, obs_mean, yerr=obs_se,
                color=C_OBS, lw=1.8, marker='o', markersize=5,
                capsize=3, capthick=1.2, elinewidth=1.0,
                label='Observed', zorder=3)

    ax.errorbar(hops, syn_mean, yerr=syn_se,
                color=C_SYN, lw=1.8, marker='s', markersize=5,
                linestyle='--', capsize=3, capthick=1.2, elinewidth=1.0,
                label='Synthetic', zorder=3)

    # ── axes styling ──────────────────────────────────────
    ax.set_title(title, fontsize=10, pad=6)
    ax.set_xlabel('Hops to NZ', fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_xticks(hops)
    ax.tick_params(axis='both', labelsize=8)
    ax.grid(True, linestyle='--', alpha=0.45, linewidth=0.6)
    ax.set_facecolor('white')

    # ── y-axis format ─────────────────────────────────────
    if col == 'p_estab':
        # always use scientific notation for P_estab
        ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
        ax.yaxis.get_major_formatter().set_powerlimits((-4, -4))
        ax.ticklabel_format(axis='y', style='sci', scilimits=(-4, -4))
    elif col == 'p_alien':
        ax.set_ylim(0.99, 1.001)
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.4f'))
    else:
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))

    ax.legend(fontsize=8, framealpha=0.85, loc='best')

    # ── add sample-count annotation (bottom-right) ────────
    n_obs_total = int(sum(len(obs[obs['hops_to_nz'] == h]) for h in hops))
    n_syn_total = int(sum(len(syn[syn['hops_to_nz'] == h]) for h in hops))
    ax.text(0.98, 0.04,
            f'n obs={n_obs_total:,}\nn syn={n_syn_total:,}',
            transform=ax.transAxes, fontsize=6.5,
            ha='right', va='bottom', color='#555555',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.75))


plt.tight_layout(rect=[0, 0, 1, 0.97])

# ── Save (BytesIO → file, avoids WPS lock) ────────────────
for ext in ('png', 'pdf'):
    buf = BytesIO()
    plt.savefig(buf, format=ext, dpi=200, bbox_inches='tight')
    buf.seek(0)
    out_path = FIG_DIR / f'track1_filter_component_by_hop.{ext}'
    with open(out_path, 'wb') as f:
        f.write(buf.read())
    print(f'       Saved → {out_path}')

plt.close()

# ══════════════════════════════════════════════════════════
# 5. Print numerical summary table
# ══════════════════════════════════════════════════════════

print('\n[Table] Hop-wise mean values:')
print(f'  {"Hop":>4}  {"Pa_obs":>10}  {"Pa_syn":>10}  '
      f'{"Pi_obs":>10}  {"Pi_syn":>10}  '
      f'{"Pe_obs (×1e4)":>14}  {"Pe_syn (×1e4)":>14}')
print('  ' + '─' * 84)
for h in hops:
    pa_o = _get(stats_obs, 'p_alien', h, 'mean', 1.0)
    pa_s = _get(stats_syn, 'p_alien', h, 'mean', 1.0)
    pi_o = _get(stats_obs, 'p_intro', h, 'mean', 1.0)
    pi_s = _get(stats_syn, 'p_intro', h, 'mean', 1.0)
    pe_o = _get(stats_obs, 'p_estab', h, 'mean', 1e4)
    pe_s = _get(stats_syn, 'p_estab', h, 'mean', 1e4)
    print(f'  {h:>4}  {pa_o:>10.4f}  {pa_s:>10.4f}  '
          f'{pi_o:>10.4f}  {pi_s:>10.4f}  '
          f'{pe_o:>14.4f}  {pe_s:>14.4f}')

print('\nDone.')