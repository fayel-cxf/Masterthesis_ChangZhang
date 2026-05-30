"""
risk_visualization_layer3.py
 
**run after risk_calcultion_mc.py
=============================
Track 1 Markov Chain — Risk Distribution Visualization + Layer 3 Verification
 
All chart text uses English, code comments use English.
 
Chart Content:
  Figure 1 — Overall Risk Distribution Comparison (KDE + Histogram + CDF)
  Figure 2 — Risk Distribution Decomposed by Cluster
  Figure 3 — Mean Risk Comparison Decomposed by Vessel Type
  Figure 4 — Gap (ii) Hop-by-Hop Risk Decreasing Structure
 
Layer 3 Statistical Tests:
  Gap (i)  — K-S Test: Whether the right tail of synthetic risk distribution is wider than observed
  Gap (ii) — Spearman Monotonicity Test: Whether hop-by-hop risk decreases systematically
"""
 
import matplotlib
matplotlib.use('Agg')   # Force non-interactive backend to ensure savefig re-renders from scratch each time

import ast
import re
import warnings
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import gaussian_kde, ks_2samp, spearmanr, mannwhitneyu

warnings.filterwarnings('ignore')
plt.close('all')   # Clean up residual figure objects from previous runs


def save_fig(path: Path, dpi: int = 150, **kwargs):
    """
    Save the current figure to disk.

    First render to a memory buffer (BytesIO), then write to disk using Python's 
    native open(), completely bypassing matplotlib's file I/O. Even if the target 
    file is locked by WPS / Image Viewer (read-only lock), the write will complete successfully.
    """
    buf = BytesIO()
    plt.savefig(buf, dpi=dpi, bbox_inches='tight', facecolor='white', **kwargs)
    plt.close()
    buf.seek(0)
    Path(path).write_bytes(buf.read())
    print(f'       Saved: {Path(path).name}')
 
# ══════════════════════════════════════════════════════════
# 0. Path Configuration
# ══════════════════════════════════════════════════════════
 
SCRIPT_DIR   = Path(__file__).resolve().parent
BASE_DIR     = SCRIPT_DIR.parent.parent.parent   # → track1_markovchain/
RESULTS_PATH = BASE_DIR / 'data/seebens/track1_risk_results.csv'
OUT_DIR      = BASE_DIR / 'data/seebens/figures'
OUT_DIR.mkdir(parents=True, exist_ok=True)
 
# ── Color Palette ─────────────────────────────────────────────
C_OBS      = '#2E86AB'   # Observed — Blue
C_SYN      = '#E84855'   # Synthetic — Red
C_GRID     = '#E8E8E8'
ALPHA_FILL = 0.35
 
# ══════════════════════════════════════════════════════════
# 1. Load Data
# ══════════════════════════════════════════════════════════
 
print('[Load] Reading risk results...')
df = pd.read_csv(RESULTS_PATH)
 
# Parse the list field of risk per port
def safe_eval(val):
    if pd.isna(val):
        return []
    s = re.sub(r'np\.\w+\(([^)]+)\)', r'\1', str(val))
    try:
        return ast.literal_eval(s)
    except Exception:
        return []
 
df['pr_inv_per_port'] = df['pr_inv_per_port'].apply(safe_eval)
 
obs = df[df['source'] == 'observed'].copy()
syn = df[df['source'] == 'synthetic'].copy()
 
obs_p = obs['p_invasion'].values
syn_p = syn['p_invasion'].values
 
print(f'       Observed routes: {len(obs):,}  Synthetic routes: {len(syn):,}')
 
# ══════════════════════════════════════════════════════════
# 2. Helper Functions
# ══════════════════════════════════════════════════════════
 
def style_ax(ax, title='', xlabel='', ylabel=''):
    """Unify chart style"""
    ax.set_facecolor('#FAFAFA')
    ax.grid(True, color=C_GRID, linewidth=0.8, zorder=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#CCCCCC')
    ax.spines['bottom'].set_color('#CCCCCC')
    if title:
        ax.set_title(title, fontsize=10, fontweight='bold', pad=8)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9)
    ax.tick_params(labelsize=8)
 
 
def pval_str(p):
    """Format p-value display"""
    if p < 0.001:
        return 'p < 0.001'
    return f'p = {p:.3f}'
 
 
def frechet_distance_1d(a, b, n_points=500):
    """
    Approximate Fréchet distance (L-infinity distance) between 1D empirical CDFs.
    Mathematically equivalent to K-S statistic D, naming kept consistent with Track 2.
    """
    lo = min(a.min(), b.min())
    hi = max(a.max(), b.max())
    xs   = np.linspace(lo, hi, n_points)
    cdf_a = np.searchsorted(np.sort(a), xs, side='right') / len(a)
    cdf_b = np.searchsorted(np.sort(b), xs, side='right') / len(b)
    return float(np.max(np.abs(cdf_a - cdf_b)))
 
 
# ══════════════════════════════════════════════════════════
# 3. Layer 3 Statistical Tests
# ══════════════════════════════════════════════════════════
 
print('\n[Layer 3] Running statistical tests...')
 
# ── Gap (i)：K-S Test, focusing on the right tail ──────────────────────────
# Two-sample K-S test: whether the overall distributions of synthetic and observed are different
ks_stat, ks_p = ks_2samp(obs_p, syn_p, alternative='two-sided')
 
# One-sided K-S test (alternative='less'): test whether synthetic is systematically larger than observed
# H0: syn <= obs; H1: syn > obs (synthetic has a heavier right tail)
ks_stat_rt, ks_p_rt = ks_2samp(obs_p, syn_p, alternative='less')
 
# Fréchet Distance (naming kept consistent with Track 2)
fd           = frechet_distance_1d(obs_p, syn_p)
fd_threshold = 0.15
 
# Specific right-tail metrics
obs_p90 = np.percentile(obs_p, 90)
obs_p95 = np.percentile(obs_p, 95)
obs_p99 = np.percentile(obs_p, 99)
syn_p90 = np.percentile(syn_p, 90)
syn_p95 = np.percentile(syn_p, 95)
syn_p99 = np.percentile(syn_p, 99)
obs_max = obs_p.max()
syn_max = syn_p.max()
tail_extension_pct = (syn_max - obs_max) / obs_max * 100
 
# ── Added: Comprehensive right-tail metrics ────────────────────────────────────
# Metric 1: tail mass ratio — proportion of voyages in synthetic exceeding observed P90 / 10%
# Meaning: the coverage density of synthetic distribution over the high-risk interval (1.0 = identical to observed)
tail_mass_syn   = float((syn_p > obs_p90).mean())   # Proportion of synthetic > obs_P90
tail_mass_obs   = 0.10                               # Observed > obs_P90 = 10% by definition
tail_mass_ratio = tail_mass_syn / tail_mass_obs
 
# Metric 2: P95 ratio — synthetic P95 / observed P95
p95_ratio = syn_p95 / obs_p95
 
# Metric 3: P99 ratio — synthetic P99 / observed P99
p99_ratio = syn_p99 / obs_p99
 
# ── Mean comparison of three Filters: identify the drivers of risk discrepancy ──────────
# Ensure all three component columns are parsed into numerical lists (pr_inv_per_port is parsed at load, adding the rest here)
for col in ['p_alien_per_port', 'p_intro_per_port', 'p_estab_per_port']:
    if not isinstance(df[col].iloc[0], list):
        df[col] = df[col].apply(safe_eval)
    # safe_eval might return string elements, force converting to float
    df[col] = df[col].apply(lambda lst: [float(x) for x in lst] if lst else [])
obs = df[df['source'] == 'observed'].copy()
syn = df[df['source'] == 'synthetic'].copy()
 
def filter_mean(grp, col):
    """Flatten the port-by-port list and calculate the mean across all ports"""
    return np.mean([v for row in grp[col] for v in row])
 
pa_obs = filter_mean(obs, 'p_alien_per_port')
pi_obs = filter_mean(obs, 'p_intro_per_port')
pe_obs = filter_mean(obs, 'p_estab_per_port')
 
pa_syn = filter_mean(syn, 'p_alien_per_port')
pi_syn = filter_mean(syn, 'p_intro_per_port')
pe_syn = filter_mean(syn, 'p_estab_per_port')
 
print('\n  Mean comparison of three Filters (Port-by-Port Average):')
print(f'  {"Filter":20} {"Observed":>12} {"Synthetic":>12} {"Difference":>10}')
print('  ' + '-'*58)
print(f'  {"P_alien":20} {pa_obs:>12.4f} {pa_syn:>12.4f} '
      f'{(pa_syn-pa_obs)/pa_obs*100:>+9.1f}%')
print(f'  {"P_intro":20} {pi_obs:>12.6f} {pi_syn:>12.6f} '
      f'{(pi_syn-pi_obs)/pi_obs*100:>+9.1f}%')
print(f'  {"P_estab (x1e6)":20} {pe_obs*1e6:>12.4f} {pe_syn*1e6:>12.4f} '
      f'{(pe_syn-pe_obs)/pe_obs*100:>+9.1f}%')
print(f'\n  → The filter with the largest difference is the primary driver of lower risk means')
 
print(f'\n  Gap (i) — Right-Tail Extension Test:')
print(f'    Fréchet Distance:    {fd:.4f}  '
      f'({"PASS" if fd < fd_threshold else "FAIL"}, threshold < {fd_threshold})')
print(f'    K-S Statistic (two-sided):  D = {ks_stat:.4f},  {pval_str(ks_p)}')
print(f'    K-S Statistic (right-tail):  D = {ks_stat_rt:.4f},  {pval_str(ks_p_rt)}')
print(f'    P90 obs / syn:  {obs_p90*1e4:.4f} / {syn_p90*1e4:.4f} ×10⁻⁴')
print(f'    P95 obs / syn:  {obs_p95*1e4:.4f} / {syn_p95*1e4:.4f} ×10⁻⁴  (ratio={p95_ratio:.3f})')
print(f'    P99 obs / syn:  {obs_p99*1e4:.4f} / {syn_p99*1e4:.4f} ×10⁻⁴  (ratio={p99_ratio:.3f})')
print(f'    Max Comparison: Observed = {obs_max*1e4:.4f}×10⁻⁴,  Synthetic = {syn_max*1e4:.4f}×10⁻⁴  ({tail_extension_pct:+.2f}%)')
print(f'    Tail mass ratio (syn>obs_P90): {tail_mass_syn*100:.1f}% / 10.0%  = {tail_mass_ratio:.3f}')
 
# Three-criteria evaluation (calculated prior to printing)
_crit1 = tail_mass_ratio >= 0.70
_crit2 = p95_ratio        >= 0.80
_crit3 = syn_max           > obs_max
print(f'    Three-criteria evaluation: crit1(TMR>=0.70)={"✓" if _crit1 else "✗"}  '
      f'crit2(P95>=0.80)={"✓" if _crit2 else "✗"}  '
      f'crit3(max>obs)={"✓" if _crit3 else "✗"}')
 
# ── Gap (ii)：Spearman Monotonicity Test ──────────────────────
# Flatten port-by-port Pr(Inv)_i and log the hop counts
records = []
for _, row in syn.iterrows():
    pr_list = row['pr_inv_per_port']
    n_src   = len(pr_list)
    for pos, pr in enumerate(pr_list):
        hops = n_src - pos    # 1 = source port closest to NZ, larger values indicate further away
        records.append({'hops_to_nz': hops, 'pr_inv': float(pr)})
 
pr_df = pd.DataFrame(records)
# ── Hop-by-hop flattening for observed data (Added) ──────────────────────────────
obs_records = []
for _, row in obs.iterrows():
    pr_list = row['pr_inv_per_port']
    n_src   = len(pr_list)
    for pos, pr in enumerate(pr_list):
        hops = n_src - pos
        obs_records.append({'hops_to_nz': hops, 'pr_inv': float(pr)})

obs_pr_df = pd.DataFrame(obs_records)

# Calculate means for each hop count
hop_counts  = pr_df['hops_to_nz'].value_counts()
valid_hops  = sorted([h for h in hop_counts.index if hop_counts[h] >= 30 and h <= 8])
hop_means   = [pr_df[pr_df['hops_to_nz'] == h]['pr_inv'].mean() for h in valid_hops]
 
# Spearman Correlation: