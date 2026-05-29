"""
Track 2 CVAE — Data Preprocessing
Builds vocab, encodes sequences, constructs condition vectors.
"""
import ast
import pickle
import numpy as np
import pandas as pd
from collections import Counter

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (CLUSTER_CSV, MAX_LEN, MIN_FREQ,
                    VESSEL_TYPES, NZ_PORTS, CONDITION_DIM, PORT_NAME_MAP)


# ── helpers ───────────────────────────────────────────────────────────────────

def safe_list(val):
    """Parse a stringified Python list."""
    if isinstance(val, list):
        return val
    try:
        return ast.literal_eval(val)
    except Exception:
        return []


def load_data():
    df = pd.read_csv(CLUSTER_CSV, encoding='latin-1')

    # Normalise cluster column
    if 'cluster' in df.columns and 'cluster_k' not in df.columns:
        df = df.rename(columns={'cluster': 'cluster_k'})

    for col in ['ports', 'transits_hours', 'dwells_hours']:
        df[col] = df[col].apply(safe_list)

    # DWT: prefer 'DWT' column
    if 'DWT' in df.columns:
        df['dwt'] = pd.to_numeric(df['DWT'], errors='coerce').fillna(10000.0)
    else:
        df['dwt'] = 10000.0

    # nz_destination
    if 'nz_dest_port' in df.columns and 'nz_destination' not in df.columns:
        df['nz_destination'] = df['nz_dest_port']
    df['nz_destination'] = df['nz_destination'].str.upper().str.strip()

    # ── AIS port name normalisation ───────────────────────────────────────────
    # Standardise naming variants to canonical port names before any filtering
    # or vocab construction.  PORT_NAME_MAP is defined in config.py.
    def normalise_port(name):
        key = name.upper().strip()
        return PORT_NAME_MAP.get(key, key)

    df['ports'] = df['ports'].apply(
        lambda ps: [normalise_port(p) for p in ps])
    df['nz_destination'] = df['nz_destination'].apply(
        lambda p: PORT_NAME_MAP.get(p, p))

    # Keep only rows whose last port is a canonical NZ destination port.
    # Industrial terminals (Tiwai Point, Marsden Point, etc.) and Picton
    # are implicitly excluded here because they are not in NZ_PORTS.
    df['last_port'] = df['ports'].apply(
        lambda p: p[-1] if p else '')
    df = df[df['last_port'].isin(NZ_PORTS)].copy()
    df = df[df['n_ports'] >= 5].copy()
    df = df.reset_index(drop=True)
    print(f"Loaded {len(df)} voyages after filtering.")
    return df


# ── vocab ─────────────────────────────────────────────────────────────────────

def build_vocab(df):
    port_freq = Counter(
        p.upper().strip()
        for row in df['ports']
        for p in row
    )
    n_rare = sum(1 for f in port_freq.values() if f < MIN_FREQ)
    print(f"Total unique ports: {len(port_freq)} | rare (<{MIN_FREQ}): {n_rare}")

    special = ['<PAD>', '<END>', '<UNK>']
    freq_ports = sorted(p for p, c in port_freq.items() if c >= MIN_FREQ)
    vocab = special + freq_ports

    port2idx = {p: i for i, p in enumerate(vocab)}
    idx2port = {i: p for p, i in port2idx.items()}

    PAD_IDX = port2idx['<PAD>']
    END_IDX = port2idx['<END>']
    UNK_IDX = port2idx['<UNK>']
    VOCAB_SIZE = len(vocab)

    print(f"Vocab size: {VOCAB_SIZE}  (PAD={PAD_IDX}, END={END_IDX}, UNK={UNK_IDX})")
    return port2idx, idx2port, PAD_IDX, END_IDX, UNK_IDX, VOCAB_SIZE


def get_port_idx(port_name, port2idx, UNK_IDX):
    key = port_name.upper().strip()
    return port2idx.get(key, UNK_IDX)


# ── time normalisation ────────────────────────────────────────────────────────

def compute_time_stats(df):
    all_tr = [t for row in df['transits_hours'] for t in row if t > 0]
    all_dw = [d for row in df['dwells_hours']   for d in row if d > 0]
    MAX_TRANSIT = float(np.quantile(all_tr, 0.99))
    MAX_DWELL   = float(np.quantile(all_dw, 0.99))
    P99_DWELL   = MAX_DWELL
    print(f"MAX_TRANSIT (p99): {MAX_TRANSIT:.1f} h | MAX_DWELL (p99): {MAX_DWELL:.1f} h")
    return MAX_TRANSIT, MAX_DWELL, P99_DWELL


def log_normalize(x, max_val):
    return np.log(np.clip(x, 0, max_val) + 1) / np.log(max_val + 1)


def log_denormalize(x_norm, max_val):
    return np.exp(x_norm * np.log(max_val + 1)) - 1


# ── port coordinates ──────────────────────────────────────────────────────────

def extract_port_coords(df):
    """
    Build {PORT_NAME: (lat, lon)} from latitudes/longitudes columns.
    Each voyage stores parallel lists of port names and coordinates.
    """
    coord_acc = {}
    for _, row in df.iterrows():
        ports = [p.upper().strip() for p in row['ports']]
        lats  = safe_list(row['latitudes'])  if isinstance(row.get('latitudes'), str) else row.get('latitudes', [])
        lons  = safe_list(row['longitudes']) if isinstance(row.get('longitudes'), str) else row.get('longitudes', [])
        if not isinstance(lats, list): lats = safe_list(str(lats))
        if not isinstance(lons, list): lons = safe_list(str(lons))
        for p, lat, lon in zip(ports, lats, lons):
            if p not in coord_acc:
                coord_acc[p] = []
            try:
                coord_acc[p].append((float(lat), float(lon)))
            except Exception:
                pass
    port_coords = {p: (np.mean([c[0] for c in cs]),
                       np.mean([c[1] for c in cs]))
                   for p, cs in coord_acc.items() if cs}
    print(f"Extracted coordinates for {len(port_coords)} ports.")
    return port_coords


# ── condition vector ──────────────────────────────────────────────────────────

def build_condition_vector(row, vessel_type_map, nz_port_map):
    """Return a 24-dim numpy condition vector: [vessel_type 8d | dwt 1d | cluster 5d | nz_port 10d]."""
    vtype = row['nbic_type_group']
    if vtype not in vessel_type_map:
        vtype = 'Other'
    type_oh = np.zeros(len(VESSEL_TYPES))
    type_oh[vessel_type_map[vtype]] = 1.0

    dwt_norm = np.array([min(float(row['dwt']) / 200000.0, 1.0)])

    cluster_oh = np.zeros(5)
    ck = int(row['cluster_k']) if not pd.isna(row['cluster_k']) else 0
    cluster_oh[ck % 5] = 1.0

    nz_dest = row['nz_destination'].upper().strip()
    if nz_dest not in nz_port_map:
        nz_dest = NZ_PORTS[0]
    nz_oh = np.zeros(len(NZ_PORTS))
    nz_oh[nz_port_map[nz_dest]] = 1.0

    return np.concatenate([type_oh, dwt_norm, cluster_oh, nz_oh]).astype(np.float32)


# ── sequence encoding ─────────────────────────────────────────────────────────

def encode_voyage(row, port2idx, PAD_IDX, END_IDX, UNK_IDX,
                  MAX_TRANSIT, MAX_DWELL):
    import torch
    ports    = [p.upper().strip() for p in row['ports']]
    transits = row['transits_hours']
    dwells   = row['dwells_hours']
    n        = row['n_ports']

    port_indices = []
    time_feats   = []

    for t in range(n):
        port_indices.append(get_port_idx(ports[t], port2idx, UNK_IDX))
        tr = log_normalize(transits[t], MAX_TRANSIT)
        dw = log_normalize(dwells[t],   MAX_DWELL)
        time_feats.append([tr, dw])

    # pad
    while len(port_indices) < MAX_LEN:
        port_indices.append(PAD_IDX)
        time_feats.append([0.0, 0.0])

    mask = [1.0] * n + [0.0] * (MAX_LEN - n)

    return (
        torch.tensor(port_indices, dtype=torch.long),
        torch.tensor(time_feats,   dtype=torch.float32),
        torch.tensor(mask,         dtype=torch.float32),
    )


# ── full preprocessing ────────────────────────────────────────────────────────

def preprocess(save_pkl=True):
    df = load_data()

    port2idx, idx2port, PAD_IDX, END_IDX, UNK_IDX, VOCAB_SIZE = build_vocab(df)
    MAX_TRANSIT, MAX_DWELL, P99_DWELL = compute_time_stats(df)
    port_coords = extract_port_coords(df)

    vessel_type_map = {v: i for i, v in enumerate(VESSEL_TYPES)}
    nz_port_map     = {p: i for i, p in enumerate(NZ_PORTS)}

    artefacts = dict(
        port2idx=port2idx, idx2port=idx2port,
        PAD_IDX=PAD_IDX, END_IDX=END_IDX, UNK_IDX=UNK_IDX,
        VOCAB_SIZE=VOCAB_SIZE,
        MAX_TRANSIT=MAX_TRANSIT, MAX_DWELL=MAX_DWELL, P99_DWELL=P99_DWELL,
        port_coords=port_coords,
        vessel_type_map=vessel_type_map, nz_port_map=nz_port_map,
    )

    if save_pkl:
        pkl_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                'preprocessed_artefacts.pkl')
        with open(pkl_path, 'wb') as f:
            pickle.dump(artefacts, f)
        print(f"Artefacts saved to {pkl_path}")

    return df, artefacts


if __name__ == '__main__':
    preprocess()
