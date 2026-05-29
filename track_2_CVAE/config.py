"""
Track 2 CVAE — Global Configuration
"""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(BASE_DIR, "..", "data")
OUT_DIR   = os.path.join(BASE_DIR, "..", "output")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Data paths ────────────────────────────────────────────────────────────────
CLUSTER_CSV  = os.path.join(DATA_DIR, "cluster.m1.labels.k5.csv")
SEA_DIST_PKL = os.path.join(DATA_DIR, "sea_distance_matrix.pkl")

# ── Sequence ──────────────────────────────────────────────────────────────────
MAX_LEN       = 7
MIN_FREQ      = 3     # ports appearing < MIN_FREQ times → <UNK>

# ── Condition vector ──────────────────────────────────────────────────────────
VESSEL_TYPES = ['Bulker', 'Container', 'General Cargo', 'Other',
                'Passenger', 'Reefer', 'RoRo', 'Tanker']

# 10 canonical NZ destination ports (expanded from 8).
# PORT CHALMERS replaces DUNEDIN/CHALMERS; GISBORNE added.
NZ_PORTS = ['AUCKLAND', 'TAURANGA', 'LYTTELTON', 'NAPIER',
            'NEW PLYMOUTH', 'NELSON', 'TIMARU', 'WELLINGTON',
            'PORT CHALMERS', 'GISBORNE']

# AIS naming variants → canonical port name.
# Applied to every port name in voyage sequences AND nz_dest_port before
# any filtering or vocab construction.
PORT_NAME_MAP = {
    'HAWKE BAY':  'NAPIER',         # anchorage area recorded as separate port
    'DUNEDIN':    'PORT CHALMERS',  # city name used instead of terminal name
    'CHALMERS':   'PORT CHALMERS',  # abbreviated form
    'TARANAKI':   'NEW PLYMOUTH',   # offshore/terminal identifier for same port
}

# ── Model ─────────────────────────────────────────────────────────────────────
EMBED_DIM     = 32
HIDDEN_DIM    = 256
LATENT_DIM    = 32
CONDITION_DIM = 24   # 8 vessel + 1 dwt + 5 cluster + 10 nz_port
GRU_LAYERS    = 2
GRU_DROPOUT   = 0.3
EMBED_DROPOUT = 0.1

# ── Training ──────────────────────────────────────────────────────────────────
LR            = 1e-3
WEIGHT_DECAY  = 1e-4
BATCH_SIZE    = 64
TOTAL_EPOCHS  = 100
KL_WARMUP     = 0.3
KL_BETA_MAX   = 1.5
GRAD_CLIP     = 1.0
LAMBDA1       = 0.3
LAMBDA2       = 0.3
VAL_SPLIT     = 0.1   # 10% held out for validation monitoring

# ── Generation ────────────────────────────────────────────────────────────────
TEMPERATURE   = 1.2
TOP_P         = 0.9
MAX_ATTEMPTS  = 5     # per voyage generation attempt

# ── Layer 2 filter ────────────────────────────────────────────────────────────
LAYER2_MIN_DWELL   = 6     # hours
LAYER2_MAX_DAYS    = 60
CIRCUITY_FACTOR    = 1.3

# Vessel speed ranges (knots): (v_min=p5, v_max=p95×1.05)
# Derived empirically from observed AIS leg speeds (dist_nm / transit_h),
# after filtering artefacts outside [0.5, 30.0] kt and legs < 50 nm.
VESSEL_SPEED = {
    'Bulker':        (3.6, 14.8),
    'Container':     (4.6, 19.4),
    'General Cargo': (4.0, 18.3),
    'Other':         (6.8, 14.3),
    'Passenger':     (6.1, 20.4),
    'Reefer':        (7.5, 20.9),
    'RoRo':          (5.7, 19.1),
    'Tanker':        (4.5, 14.4),
}
