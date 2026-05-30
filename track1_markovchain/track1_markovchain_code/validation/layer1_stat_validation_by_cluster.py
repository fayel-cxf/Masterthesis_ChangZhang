"""
mc_validation_by_cluster.py
============================
Track 1 Markov Chain Synthetic Voyage Validation (per cluster)

Compares synthetic and observed voyages independently for each cluster,
avoiding cross-cluster mixing that could obscure real distributional issues.

Outputs:
  data/markov/validation.result/mc_validation_cluster{k}_report.txt
  data/markov/validation.result/mc_validation_cluster{k}_plots.png
  data/markov/validation.result/mc_validation_summary.txt
"""

import ast
import re
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp

warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════
# 0. Path configuration
# ══════════════════════════════════════════════
SCRIPT_DIR  = Path(__file__).resolve().parent
BASE_DIR    = SCRIPT_DIR.parent.parent.parent   # → track1_markovchain/
OBS_PATH    = BASE_DIR / 'data/clustering/cluster.m1.labels.k5.csv'
SYN_PATH    = BASE_DIR / 'data/markov/output/mc.syn.all.csv'
OUTPUT_DIR  = BASE_DIR / 'data/markov/validation.result/mc.validation'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

COLOR_OBS = '#2E86AB'
COLOR_SYN = '#E84855'
ALPHA_BAR = 0.65

# ══════════════════════════════════════════════
# 1. Data loading
# ══════════════════════════════════════════════

def safe_eval(val):
    """Parse list fields from CSV, compatible with np.str_ and other numpy expressions"""
    if pd.isna(val):
        return []
    s = str(val)
    s = re.sub(r'np\.\w+\(([^)]+)\)', r'\1', s)
    try:
        return ast.literal_eval(s)
    except Exception:
        return []


def load_data(obs_path: Path, syn_path: Path):
    """Load and parse observed and synthetic data, compute derived fields"""
    print("[Load] Observed data...")
    df_obs = pd.read_csv(obs_path)

    print("[Load] Synthetic data...")
    df_syn = pd.read_csv(syn_path)

    list_cols = ['ports', 'port_territories', 'latitudes', 'longitudes',
                 'ecoprovinces', 'dwells_hours', 'transits_hours']

    for col in list_cols:
        if col in df_obs.columns:
            df_obs[col] = df_obs[col].apply(ast.literal_eval)
        if col in df_syn.columns:
            df_syn[col] = df_syn[col].apply(safe_eval)

    # Compute delta_t for observed data
    def compute_delta_t(row):
        t = row['transits_hours']
        d = row['dwells_hours']
        n = row['n_ports']
        return sum(t[1:]) + sum(d[1:n-1])

    df_obs['delta_t_hours'] = df_obs.apply(compute_delta_t, axis=1)

    print(f"       Observed: {len(df_obs)} voyages,  Synthetic: {len(df_syn)} voyages")
    print(f"       Observed cluster distribution:\n{df_obs['cluster'].value_counts().sort_index().to_string()}")
    print(f"       Synthetic cluster distribution:\n{df_syn['cluster'].value_counts().sort_index().to_string()}")

    return df_obs, df_syn


# ══════════════════════════════════════════════
# 2. Metrics (per cluster)
# ══════════════════════════════════════════════

def compute_metrics(obs: pd.DataFrame, syn: pd.DataFrame) -> dict:
    """Compute all validation metrics for a single cluster"""
    results = {}

    # -- Metric 1: Port visit frequency JSD --
    obs_ports = [p for ports in obs['ports'] for p in ports]
    syn_ports = [p for ports in syn['ports'] for p in ports]
    all_ports = sorted(set(obs_ports) | set(syn_ports))
    obs_cnt = pd.Series(obs_ports).value_counts()
    syn_cnt = pd.Series(syn_ports).value_counts()
    obs_f = np.array([obs_cnt.get(p, 0) for p in all_ports], dtype=float)
    syn_f = np.array([syn_cnt.get(p, 0) for p in all_ports], dtype=float)
    obs_f /= obs_f.sum()
    syn_f /= syn_f.sum()
    jsd = float(jensenshannon(obs_f, syn_f) ** 2)
    # top20 for visualisation
    top20 = pd.Series(obs_ports).value_counts().head(20).index.tolist()
    results['port_freq'] = {
        'jsd': jsd, 'passed': jsd < 0.05,
        'all_ports': all_ports, 'obs_freq': obs_f, 'syn_freq': syn_f,
        'top20': top20,
        'obs_top': np.array([obs_cnt.get(p,0) for p in top20], dtype=float),
        'syn_top': np.array([syn_cnt.get(p,0) for p in top20], dtype=float),
    }

    # -- Metric 2: n_ports KS --
    stat, pval = ks_2samp(obs['n_ports'].values.astype(float),
                          syn['n_ports'].values.astype(float))
    obs_nd = obs['n_ports'].value_counts(normalize=True).sort_index()
    syn_nd = syn['n_ports'].value_counts(normalize=True).sort_index()
    results['nports'] = {
        'ks_stat': stat, 'p_value': pval, 'passed': pval > 0.05,
        'obs_dist': obs_nd, 'syn_dist': syn_nd,
        'all_vals': sorted(set(obs_nd.index) | set(syn_nd.index)),
    }

    # -- Metric 3: per-leg transit KS --
    obs_t = [t for row in obs.itertuples() for t in row.transits_hours[1:] if t > 0]
    syn_t = [t for row in syn.itertuples() for t in row.transits_hours[1:] if t > 0]
    stat, pval = ks_2samp(obs_t, syn_t)
    results['transit'] = {
        'ks_stat': stat, 'p_value': pval, 'passed': pval > 0.05,
        'obs_vals': np.array(obs_t), 'syn_vals': np.array(syn_t),
    }

    # -- Metric 4: dwell KS --
    def get_dwells(df):
        return [d for row in df.itertuples()
                for d in row.dwells_hours[1:row.n_ports-1] if d > 0]
    obs_d = get_dwells(obs)
    syn_d = get_dwells(syn)
    stat, pval = ks_2samp(obs_d, syn_d)
    results['dwell'] = {
        'ks_stat': stat, 'p_value': pval, 'passed': pval > 0.05,
        'obs_vals': np.array(obs_d), 'syn_vals': np.array(syn_d),
    }

    # -- Metric 5: total span days --
    obs_s = obs['total_span_days'].values
    syn_s = syn['total_span_days'].values
    stat, pval = ks_2samp(obs_s, syn_s)
    def describe(v):
        return {k: float(f(v)) for k, f in [
            ('mean', np.mean), ('median', np.median), ('std', np.std),
            ('p25', lambda x: np.percentile(x, 25)),
            ('p75', lambda x: np.percentile(x, 75)),
            ('p95', lambda x: np.percentile(x, 95)),
            ('max', np.max),
        ]}
    results['span_days'] = {
        'ks_stat': stat, 'p_value': pval, 'passed': pval > 0.05,
        'obs_stats': describe(obs_s), 'syn_stats': describe(syn_s),
        'obs_vals': obs_s, 'syn_vals': syn_s,
    }

    # -- Metric 6: vessel type JSD --
    obs_vd = obs['nbic_type_group'].value_counts(normalize=True).sort_index()
    syn_vd = syn['vessel_type'].value_counts(normalize=True).sort_index()
    all_vt = sorted(set(obs_vd.index) | set(syn_vd.index))
    ov = np.array([obs_vd.get(t, 0) for t in all_vt])
    sv = np.array([syn_vd.get(t, 0) for t in all_vt])
    jsd_vt = float(jensenshannon(ov, sv) ** 2)
    results['vessel_type'] = {
        'jsd': jsd_vt,
        'passed': jsd_vt < 0.10,
        'all_types': all_vt, 'obs_dist': obs_vd, 'syn_dist': syn_vd,
        'obs_arr': ov, 'syn_arr': sv,
    }

    # -- Metric 7: NZ destination port JSD --
    obs_nd2 = obs['nz_dest_port'].value_counts(normalize=True).sort_index()
    syn_nd2 = syn['nz_dest_port'].value_counts(normalize=True).sort_index()
    all_nz = sorted(set(obs_nd2.index) | set(syn_nd2.index))
    on = np.array([obs_nd2.get(p, 0) for p in all_nz])
    sn = np.array([syn_nd2.get(p, 0) for p in all_nz])
    jsd_nz = float(jensenshannon(on, sn) ** 2)
    results['nz_dest'] = {
        'jsd': jsd_nz,
        'passed': jsd_nz < 0.10,
        'all_ports': all_nz, 'obs_dist': obs_nd2, 'syn_dist': syn_nd2,
        'obs_arr': on, 'syn_arr': sn,
    }

    # -- Metric 8: delta-t --
    obs_dt = obs['delta_t_hours'].values
    syn_dt = syn['delta_t_hours'].values
    stat, pval = ks_2samp(obs_dt, syn_dt)
    results['delta_t'] = {
        'ks_stat': stat, 'p_value': pval, 'passed': pval > 0.05,
        'obs_stats': describe(obs_dt), 'syn_stats': describe(syn_dt),
        'obs_vals': obs_dt, 'syn_vals': syn_dt,
    }

    return results


# ══════════════════════════════════════════════
# 3. Text report (per cluster)
# ══════════════════════════════════════════════

def write_cluster_report(k: int, results: dict,
                         n_obs: int, n_syn: int,
                         path: Path) -> dict:
    """Write per-cluster report; return summary row for cross-cluster table"""
    ok  = lambda p: "PASS" if p else "FAIL"
    lines = []
    lines.append("=" * 60)
    lines.append(f"Cluster {k} Validation Report")
    lines.append("=" * 60)
    lines.append(f"Observed voyages: {n_obs}    Synthetic voyages: {n_syn}")
    lines.append("")

    r = results['port_freq']
    lines.append(f"[1] Port visit frequency JSD = {r['jsd']:.4f}   {ok(r['passed'])}")

    r = results['nports']
    lines.append(f"[2] n_ports KS p = {r['p_value']:.4f}   {ok(r['passed'])}")
    lines.append("    Observed: " + "  ".join(
        [f"n={k2}: {v*100:.1f}%" for k2, v in r['obs_dist'].items()]))
    lines.append("    Synthetic: " + "  ".join(
        [f"n={k2}: {v*100:.1f}%" for k2, v in r['syn_dist'].items()]))

    r = results['transit']
    lines.append(f"[3] Transit KS p = {r['p_value']:.4f}   {ok(r['passed'])}")
    lines.append(f"    Observed  mean={np.mean(r['obs_vals']):.1f}h  "
                 f"median={np.median(r['obs_vals']):.1f}h")
    lines.append(f"    Synthetic mean={np.mean(r['syn_vals']):.1f}h  "
                 f"median={np.median(r['syn_vals']):.1f}h")

    r = results['dwell']
    lines.append(f"[4] Dwell KS p = {r['p_value']:.4f}   {ok(r['passed'])}")
    lines.append(f"    Observed  mean={np.mean(r['obs_vals']):.1f}h  "
                 f"median={np.median(r['obs_vals']):.1f}h")
    lines.append(f"    Synthetic mean={np.mean(r['syn_vals']):.1f}h  "
                 f"median={np.median(r['syn_vals']):.1f}h")

    r = results['span_days']
    o, s = r['obs_stats'], r['syn_stats']
    lines.append(f"[5] Total span days KS p = {r['p_value']:.4f}   {ok(r['passed'])}")
    lines.append(f"    {'':10} {'Observed':>10} {'Synthetic':>10}")
    for key in ['mean', 'median', 'std', 'p25', 'p75', 'p95', 'max']:
        lines.append(f"    {key:10} {o[key]:>10.2f} {s[key]:>10.2f}")

    r = results['vessel_type']
    lines.append(f"[6] Vessel type distribution JSD = {r['jsd']:.4f}")
    lines.append(f"    {'Vessel type':20} {'Obs':>7} {'Syn':>7}")
    for vt in r['all_types']:
        lines.append(f"    {vt:20} "
                     f"{r['obs_dist'].get(vt,0)*100:>6.1f}% "
                     f"{r['syn_dist'].get(vt,0)*100:>6.1f}%")

    r = results['nz_dest']
    lines.append(f"[7] NZ destination port distribution JSD = {r['jsd']:.4f}")
    lines.append(f"    {'Port':20} {'Obs':>7} {'Syn':>7}")
    for p in r['all_ports']:
        lines.append(f"    {p:20} "
                     f"{r['obs_dist'].get(p,0)*100:>6.1f}% "
                     f"{r['syn_dist'].get(p,0)*100:>6.1f}%")

    r = results['delta_t']
    o, s = r['obs_stats'], r['syn_stats']
    lines.append(f"[8] Delta-t KS p = {r['p_value']:.4f}   {ok(r['passed'])}")
    lines.append(f"    {'':10} {'Observed':>10} {'Synthetic':>10}")
    for key in ['mean', 'median', 'std', 'p25', 'p75', 'p95']:
        lines.append(f"    {key:10} {o[key]:>10.2f} {s[key]:>10.2f}")

    text = "\n".join(lines)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text)

    # Return row for summary table
    return {
        'cluster':          k,
        'n_obs':            n_obs,
        'n_syn':            n_syn,
        'port_jsd':         results['port_freq']['jsd'],
        'port_pass':        results['port_freq']['passed'],
        'nports_p':         results['nports']['p_value'],
        'nports_d':         results['nports']['ks_stat'],
        'nports_pass':      results['nports']['passed'],
        'transit_p':        results['transit']['p_value'],
        'transit_d':        results['transit']['ks_stat'],
        'transit_pass':     results['transit']['passed'],
        'transit_obs_med':  float(np.median(results['transit']['obs_vals'])),
        'transit_syn_med':  float(np.median(results['transit']['syn_vals'])),
        'dwell_p':          results['dwell']['p_value'],
        'dwell_d':          results['dwell']['ks_stat'],
        'dwell_pass':       results['dwell']['passed'],
        'span_obs_mean':    results['span_days']['obs_stats']['mean'],
        'span_syn_mean':    results['span_days']['syn_stats']['mean'],
        'span_obs_max':     results['span_days']['obs_stats']['max'],
        'span_syn_max':     results['span_days']['syn_stats']['max'],
        'span_p':           results['span_days']['p_value'],
        'span_d':           results['span_days']['ks_stat'],
        'span_pass':        results['span_days']['passed'],
        'vessel_jsd':       results['vessel_type']['jsd'],
        'vessel_pass':      results['vessel_type']['passed'],
        'nz_jsd':           results['nz_dest']['jsd'],
        'nz_pass':          results['nz_dest']['passed'],
        'delta_t_p':        results['delta_t']['p_value'],
        'delta_t_d':        results['delta_t']['ks_stat'],
        'delta_t_pass':     results['delta_t']['passed'],
        'delta_t_obs_med':  results['delta_t']['obs_stats']['median'],
        'delta_t_syn_med':  results['delta_t']['syn_stats']['median'],
    }


# ══════════════════════════════════════════════
# 4. Summary report
# ══════════════════════════════════════════════

def write_summary(summary_rows: list, path: Path):
    lines = []
    lines.append("=" * 80)
    lines.append("Track 1 Markov Chain Validation Summary (by Cluster)")
    lines.append("=" * 80)

    ok = lambda p: "✓" if p else "✗"

    lines.append(f"\n{'Cluster':>8} {'N_obs':>6} {'N_syn':>6} "
                 f"{'[1]JSD':>8} {'[2]nports':>10} {'[3]transit':>11} "
                 f"{'[4]dwell':>9} {'[5]span':>10} {'[6]vtype':>9} {'[7]nzdest':>10} {'[8]delta_t':>10}")
    lines.append("-" * 100)

    for r in summary_rows:
        lines.append(
            f"{r['cluster']:>8} {r['n_obs']:>6} {r['n_syn']:>6} "
            f"{r['port_jsd']:>7.4f}{ok(r['port_pass']):>2} "
            f"p={r['nports_p']:.3f}{ok(r['nports_pass']):>2} "
            f"p={r['transit_p']:.3f}{ok(r['transit_pass']):>2} "
            f"p={r['dwell_p']:.3f}{ok(r['dwell_pass']):>2} "
            f"p={r['span_p']:.3f}{ok(r['span_pass']):>2} "
            f"{r['vessel_jsd']:.4f}{ok(r['vessel_pass']):>2} "
            f"{r['nz_jsd']:.4f}{ok(r['nz_pass']):>2} "
            f"p={r['delta_t_p']:.3f}{ok(r['delta_t_pass']):>2}"
        )

    lines.append("")
    lines.append("KS D-statistic summary  (D<0.10: OK ✓  |  0.10-0.20: minor  |  >0.20: review ✗)")
    lines.append(f"{'Cluster':>8} {'[2]nports_D':>12} {'[3]transit_D':>13} "
                 f"{'[4]dwell_D':>11} {'[5]span_D':>10} {'[8]delta_t_D':>13}")
    lines.append("-" * 72)

    def d_flag(d):
        if d < 0.10:   return "✓"
        if d < 0.20:   return "~"
        return "✗"

    for r in summary_rows:
        lines.append(
            f"{r['cluster']:>8} "
            f"{r['nports_d']:>10.3f}{d_flag(r['nports_d']):>3} "
            f"{r['transit_d']:>11.3f}{d_flag(r['transit_d']):>3} "
            f"{r['dwell_d']:>9.3f}{d_flag(r['dwell_d']):>3} "
            f"{r['span_d']:>8.3f}{d_flag(r['span_d']):>3} "
            f"{r['delta_t_d']:>11.3f}{d_flag(r['delta_t_d']):>3}"
        )

    lines.append("")
    lines.append("Transit time median comparison (observed vs synthetic, hours):")
    lines.append(f"{'Cluster':>8} {'obs_median':>12} {'syn_median':>12} {'ratio':>8}")
    lines.append("-" * 45)
    for r in summary_rows:
        ratio = r['transit_syn_med'] / r['transit_obs_med'] \
                if r['transit_obs_med'] > 0 else float('nan')
        lines.append(f"{r['cluster']:>8} {r['transit_obs_med']:>12.1f} "
                     f"{r['transit_syn_med']:>12.1f} {ratio:>8.2f}x")

    lines.append("")
    lines.append("Total span days comparison (mean / max):")
    lines.append(f"{'Cluster':>8} {'obs_mean':>10} {'syn_mean':>10} "
                 f"{'obs_max':>9} {'syn_max':>9}")
    lines.append("-" * 52)
    for r in summary_rows:
        lines.append(f"{r['cluster']:>8} {r['span_obs_mean']:>10.1f} "
                     f"{r['span_syn_mean']:>10.1f} "
                     f"{r['span_obs_max']:>9.1f} {r['span_syn_max']:>9.1f}")

    lines.append("")
    lines.append("Delta-t median comparison (hours):")
    lines.append(f"{'Cluster':>8} {'obs_median':>12} {'syn_median':>12} {'ratio':>8}")
    lines.append("-" * 45)
    for r in summary_rows:
        ratio = r['delta_t_syn_med'] / r['delta_t_obs_med'] \
                if r['delta_t_obs_med'] > 0 else float('nan')
        lines.append(f"{r['cluster']:>8} {r['delta_t_obs_med']:>12.1f} "
                     f"{r['delta_t_syn_med']:>12.1f} {ratio:>8.2f}x")

    text = "\n".join(lines)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text)
    print(text)


# ══════════════════════════════════════════════
# 5. Visualisation (per cluster)
# ══════════════════════════════════════════════

def make_cluster_plot(k: int, results: dict,
                      n_obs: int, n_syn: int,
                      path: Path):
    fig = plt.figure(figsize=(18, 14))
    fig.patch.set_facecolor('#F8F9FA')
    gs = gridspec.GridSpec(2, 4, figure=fig,
                           hspace=0.45, wspace=0.35,
                           top=0.91, bottom=0.07,
                           left=0.06, right=0.97)

    title_kw = dict(fontsize=10, fontweight='bold', pad=7)
    label_kw = dict(fontsize=8)
    leg_kw   = dict(fontsize=8, framealpha=0.7)
    width    = 0.35

    def ax_style(ax, title):
        ax.set_facecolor('#FFFFFF')
        ax.set_title(title, **title_kw)
        ax.tick_params(labelsize=7.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        for sp in ['left', 'bottom']:
            ax.spines[sp].set_color('#CCCCCC')

    ok = lambda p: "✓" if p else "✗"

    # [1] Port frequency scatter plot
    ax = fig.add_subplot(gs[0, 0])
    r = results['port_freq']
    mask = (r['obs_freq'] > 0) | (r['syn_freq'] > 0)
    ax.scatter(r['obs_freq'][mask], r['syn_freq'][mask],
               alpha=0.4, s=10, color='#555555', linewidths=0)
    mv = max(r['obs_freq'][mask].max(), r['syn_freq'][mask].max())
    ax.plot([0, mv], [0, mv], 'r--', lw=1, alpha=0.6, label='y=x')
    ax_style(ax, f"[1] Port Frequency\nJSD={r['jsd']:.4f} {ok(r['passed'])}")
    ax.set_xlabel("Observed frequency", **label_kw)
    ax.set_ylabel("Synthetic frequency", **label_kw)
    ax.legend(**leg_kw)

    # [2] n_ports bar chart
    ax = fig.add_subplot(gs[0, 1])
    r = results['nports']
    vals = r['all_vals']
    x = np.arange(len(vals))
    ax.bar(x - width/2, [r['obs_dist'].get(v, 0) for v in vals],
           width, label='Observed', color=COLOR_OBS, alpha=ALPHA_BAR)
    ax.bar(x + width/2, [r['syn_dist'].get(v, 0) for v in vals],
           width, label='Synthetic', color=COLOR_SYN, alpha=ALPHA_BAR)
    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in vals])
    ax.set_ylabel("Proportion", **label_kw)
    ax_style(ax, f"[2] n_ports\nKS p={r['p_value']:.4f} {ok(r['passed'])}")
    ax.legend(**leg_kw)

    # [3] Transit time histogram
    ax = fig.add_subplot(gs[0, 2])
    r = results['transit']
    bins = np.logspace(np.log10(0.5), np.log10(1000), 35)
    ax.hist(r['obs_vals'], bins=bins, density=True,
            alpha=ALPHA_BAR, color=COLOR_OBS, label='Observed')
    ax.hist(r['syn_vals'], bins=bins, density=True,
            alpha=ALPHA_BAR, color=COLOR_SYN, label='Synthetic')
    ax.set_xscale('log')
    ax.set_xlabel("Transit time (hours)", **label_kw)
    ax.set_ylabel("Density", **label_kw)
    ax_style(ax, f"[3] Transit Time\nKS p={r['p_value']:.4f} {ok(r['passed'])}")
    ax.legend(**leg_kw)

    # [4] Dwell time histogram
    ax = fig.add_subplot(gs[0, 3])
    r = results['dwell']
    bins = np.linspace(0, 150, 30)
    ax.hist(r['obs_vals'], bins=bins, density=True,
            alpha=ALPHA_BAR, color=COLOR_OBS, label='Observed')
    ax.hist(r['syn_vals'], bins=bins, density=True,
            alpha=ALPHA_BAR, color=COLOR_SYN, label='Synthetic')
    ax.set_xlabel("Dwell time (hours)", **label_kw)
    ax.set_ylabel("Density", **label_kw)
    ax_style(ax, f"[4] Dwell Time\nKS p={r['p_value']:.4f} {ok(r['passed'])}")
    ax.legend(**leg_kw)

    # [5] Total span days histogram
    ax = fig.add_subplot(gs[1, 0])
    r = results['span_days']
    bins = np.linspace(0, max(r['obs_vals'].max(), r['syn_vals'].max()) * 1.05, 35)
    ax.hist(r['obs_vals'], bins=bins, density=True,
            alpha=ALPHA_BAR, color=COLOR_OBS, label='Observed')
    ax.hist(r['syn_vals'], bins=bins, density=True,
            alpha=ALPHA_BAR, color=COLOR_SYN, label='Synthetic')
    ax.set_xlabel("Total span (days)", **label_kw)
    ax.set_ylabel("Density", **label_kw)
    ax_style(ax, f"[5] Total Span Days\nKS p={r['p_value']:.4f} {ok(r['passed'])}")
    ax.legend(**leg_kw)

    # [6] Vessel type distribution
    ax = fig.add_subplot(gs[1, 1])
    r = results['vessel_type']
    vt = r['all_types']
    x = np.arange(len(vt))
    ax.bar(x - width/2, r['obs_arr'], width,
           label='Observed', color=COLOR_OBS, alpha=ALPHA_BAR)
    ax.bar(x + width/2, r['syn_arr'], width,
           label='Synthetic', color=COLOR_SYN, alpha=ALPHA_BAR)
    ax.set_xticks(x)
    ax.set_xticklabels(vt, fontsize=6.5, rotation=20, ha='right')
    ax.set_ylabel("Proportion", **label_kw)
    ax_style(ax, f"[6] Vessel Type\nJSD={r['jsd']:.4f} {ok(r['passed'])}")
    ax.legend(**leg_kw)

    # [7] NZ destination port distribution
    ax = fig.add_subplot(gs[1, 2])
    r = results['nz_dest']
    pts = r['all_ports']
    x = np.arange(len(pts))
    ax.bar(x - width/2, r['obs_arr'], width,
           label='Observed', color=COLOR_OBS, alpha=ALPHA_BAR)
    ax.bar(x + width/2, r['syn_arr'], width,
           label='Synthetic', color=COLOR_SYN, alpha=ALPHA_BAR)
    ax.set_xticks(x)
    ax.set_xticklabels(pts, fontsize=6, rotation=25, ha='right')
    ax.set_ylabel("Proportion", **label_kw)
    ax_style(ax, f"[7] NZ Destination Port\nJSD={r['jsd']:.4f} {ok(r['passed'])}")
    ax.legend(**leg_kw)

    # [8] Delta-t distribution
    ax = fig.add_subplot(gs[1, 3])
    r = results['delta_t']
    bins = np.linspace(0, max(r['obs_vals'].max(), r['syn_vals'].max()) * 1.05, 35)
    ax.hist(r['obs_vals'], bins=bins, density=True,
            alpha=ALPHA_BAR, color=COLOR_OBS, label='Observed')
    ax.hist(r['syn_vals'], bins=bins, density=True,
            alpha=ALPHA_BAR, color=COLOR_SYN, label='Synthetic')
    ax.set_xlabel("Delta-t (hours)", **label_kw)
    ax.set_ylabel("Density", **label_kw)
    ax_style(ax, f"[8] Delta-t\nKS p={r['p_value']:.4f} {ok(r['passed'])}")
    ax.legend(**leg_kw)

    fig.suptitle(
        f"Cluster {k} Validation  ·  Observed {n_obs} vs Synthetic {n_syn}",
        fontsize=13, fontweight='bold', y=0.97, color='#222222'
    )

    plt.savefig(path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Plot saved: {path.name}")


# ══════════════════════════════════════════════
# 6. Main
# ══════════════════════════════════════════════

def main():
    print("=" * 60)
    print("Track 1 Markov Chain Validation (by Cluster)")
    print("=" * 60)

    df_obs, df_syn = load_data(OBS_PATH, SYN_PATH)

    n_clusters   = sorted(df_obs['cluster'].unique())
    summary_rows = []

    for k in n_clusters:
        print(f"\n-- Cluster {k} --")
        obs_k = df_obs[df_obs['cluster'] == k].copy()
        syn_k = df_syn[df_syn['cluster'] == k].copy()

        if len(obs_k) == 0 or len(syn_k) == 0:
            print(f"  Skipped (observed={len(obs_k)}, synthetic={len(syn_k)})")
            continue

        results = compute_metrics(obs_k, syn_k)

        report_path = OUTPUT_DIR / f'mc_validation_cluster{k}_report.txt'
        plot_path   = OUTPUT_DIR / f'mc_validation_cluster{k}_plots.png'

        row = write_cluster_report(k, results, len(obs_k), len(syn_k), report_path)
        summary_rows.append(row)
        print(f"  Report saved: {report_path.name}")

        make_cluster_plot(k, results, len(obs_k), len(syn_k), plot_path)
        print(f"  Plot saved: {plot_path.name}")

    print("\n-- Summary --")
    summary_path = OUTPUT_DIR / 'mc_validation_summary.txt'
    write_summary(summary_rows, summary_path)
    print(f"\nSummary saved: {summary_path}")
    print("\nDone.")


if __name__ == '__main__':
    main()
