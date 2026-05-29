"""
Track 2 CVAE — Phase 1 Runner
Goal: train model, generate 5,000 valid voyages, compute QC metrics.
"""
import os, sys, pickle, time
import numpy as np
import pandas as pd
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from config import OUT_DIR, SEA_DIST_PKL
from data.preprocess       import preprocess, safe_list
from training.train        import train
from generation.filter     import (build_sea_distance_matrix,
                                   load_sea_distance_matrix,
                                   layer1_metrics,
                                   nz_destination_jsd)
from generation.sample     import generate_batch


# ─────────────────────────────────────────────────────────────────────────────
# Step 0: Preprocess & build distance matrix (one-time)
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 0 — Preprocessing & distance matrix")
print("=" * 60)

df, artefacts = preprocess(save_pkl=True)

sea_dist_nm = load_sea_distance_matrix(SEA_DIST_PKL)
if sea_dist_nm is None:
    print("Building sea-distance matrix (this takes ~5–15 min)…")
    build_sea_distance_matrix(artefacts['port_coords'], save_path=SEA_DIST_PKL)
    sea_dist_nm = load_sea_distance_matrix(SEA_DIST_PKL)
    print(f"Distance matrix built ({len(sea_dist_nm)} port pairs).")
else:
    print(f"Loaded existing distance matrix ({len(sea_dist_nm)} port pairs).")


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Train CVAE
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 1 — Training CVAE")
print("=" * 60)

model_path = os.path.join(OUT_DIR, 'best_model.pt')

if os.path.exists(model_path):
    print(f"Found existing model at {model_path}. Loading…")
    from model.cvae import VoyageCVAE
    checkpoint  = torch.load(model_path, map_location='cpu', weights_only=False)
    artefacts   = checkpoint['artefacts']   # use artefacts from checkpoint
    device      = torch.device('mps' if torch.backends.mps.is_available() else
                               'cuda' if torch.cuda.is_available() else 'cpu')
    model = VoyageCVAE(
        vocab_size  = artefacts['VOCAB_SIZE'],
        port2idx    = artefacts['port2idx'],
        port_coords = artefacts['port_coords'],
    ).to(device)
    model.load_state_dict(checkpoint['model_state'])
    print(f"Model loaded (epoch {checkpoint['epoch']}, "
          f"val_loss {checkpoint['val_loss']:.4f}).")
else:
    model, artefacts, history = train()
    device = next(model.parameters()).device


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Generate 5,000 valid voyages
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2 — Phase 1 generation (target: 5,000 valid voyages)")
print("=" * 60)

t0 = time.time()
valid_voyages = generate_batch(
    model, df, artefacts, sea_dist_nm,
    target_valid=5000,
    temperature=0.7,
    verbose=True,
    markov_alpha=2.0,
)
elapsed = time.time() - t0
print(f"Generation time: {elapsed/60:.1f} min")


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Save results
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3 — Saving results")
print("=" * 60)

out_csv  = os.path.join(OUT_DIR, 'phase1_synthetic_voyages.csv')
out_pkl  = os.path.join(OUT_DIR, 'phase1_synthetic_voyages.pkl')

rows = []
for v in valid_voyages:
    rows.append({
        'ports':          str(v['ports']),
        'transits_hours': str(v['transits_hours']),
        'dwells_hours':   str(v['dwells_hours']),
        'n_ports':        v['n_ports'],
        'vessel_type':    v.get('vessel_type', ''),
        'dwt':            v.get('dwt', np.nan),
        'cluster_k':      v.get('cluster_k', -1),
        'nz_destination': v.get('nz_destination', ''),
    })

syn_df = pd.DataFrame(rows)
syn_df.to_csv(out_csv, index=False)
with open(out_pkl, 'wb') as f:
    pickle.dump(valid_voyages, f)
print(f"Saved {len(valid_voyages)} voyages → {out_csv}")


# ─────────────────────────────────────────────────────────────────────────────
# Step 4: QC Metrics
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4 — Quality Control (Layer 1 metrics)")
print("=" * 60)

# Restore list types from observed df
df['ports'] = df['ports'].apply(
    lambda x: x if isinstance(x, list) else safe_list(x))

metrics = layer1_metrics(df, valid_voyages)
print(f"Port freq JSD:       {metrics['jsd']:.4f}  (threshold < 0.08)")
print(f"n_ports KS stat:     {metrics['ks_n_ports_stat']:.4f}")
print(f"n_ports KS p-value:  {metrics['ks_n_ports_p']:.4f}  (threshold > 0.05)")

# Gap(i): fraction of observed port transitions NOT in synthetic
print("\n--- Gap(i) estimation ---")
obs_pairs = set()
for _, row in df.iterrows():
    ports = row['ports'] if isinstance(row['ports'], list) else safe_list(row['ports'])
    for j in range(len(ports) - 1):
        obs_pairs.add((ports[j].upper(), ports[j+1].upper()))

syn_pairs = set()
for v in valid_voyages:
    for j in range(len(v['ports']) - 1):
        syn_pairs.add((v['ports'][j], v['ports'][j+1]))

new_pairs  = syn_pairs - obs_pairs
feasible   = obs_pairs | syn_pairs
gap_i      = len(new_pairs) / len(feasible) if feasible else 0.0
print(f"Observed port pairs: {len(obs_pairs)}")
print(f"Synthetic port pairs:{len(syn_pairs)}")
print(f"New (unseen) pairs:  {len(new_pairs)}")
print(f"Gap(i):              {gap_i:.2%}  (target 15–25%)")

# n_ports distribution
from collections import Counter
obs_n_dist = Counter(df['n_ports'].tolist())
syn_n_dist = Counter(v['n_ports'] for v in valid_voyages)
print("\n--- n_ports distribution ---")
print(f"{'n_ports':>8} | {'observed':>10} | {'synthetic':>10}")
for k in sorted(set(obs_n_dist) | set(syn_n_dist)):
    print(f"{k:>8} | {obs_n_dist.get(k, 0):>10} | {syn_n_dist.get(k, 0):>10}")

# NZ destination JSD (diagnostic)
print("\n--- NZ destination JSD (diagnostic) ---")
nz_metrics = nz_destination_jsd(df, valid_voyages)
nz_jsd = nz_metrics['jsd_nz_dest']
print(f"NZ destination JSD:  {nz_jsd:.4f}  (diagnostic only)")
print(f"{'port':>16} | {'observed':>10} | {'synthetic':>11} | {'obs %':>7} | {'syn %':>7}")
nz_obs_c = nz_metrics['obs_nz_counts']
nz_syn_c = nz_metrics['syn_nz_counts']
obs_total = sum(nz_obs_c.values())
for port in sorted(nz_obs_c, key=lambda p: -nz_obs_c[p]):
    o, s = nz_obs_c.get(port, 0), nz_syn_c.get(port, 0)
    print(f"{port:>16} | {o:>10} | {s:>11} | "
          f"{o/obs_total*100:>6.1f}% | {s/len(valid_voyages)*100:>6.1f}%")

# Summary
qc = {**metrics, 'gap_i': gap_i, 'jsd_nz_dest': nz_jsd, 'n_valid': len(valid_voyages)}
qc_path = os.path.join(OUT_DIR, 'phase1_qc_metrics.pkl')
with open(qc_path, 'wb') as f:
    pickle.dump(qc, f)

print("\n" + "=" * 60)
print("PHASE 1 COMPLETE")
print(f"  Valid voyages:  {len(valid_voyages)}")
print(f"  JSD:            {metrics['jsd']:.4f}  {'✓' if metrics['jsd'] < 0.08 else '✗'}")
print(f"  KS p-value:     {metrics['ks_n_ports_p']:.4f}  {'✓' if metrics['ks_n_ports_p'] > 0.05 else '✗'}")
print(f"  Gap(i):         {gap_i:.2%}  {'✓' if 0.15 <= gap_i <= 0.30 else '⚠'}")
print(f"  NZ dest JSD:    {nz_jsd:.4f}  (diagnostic)")
print("=" * 60)
