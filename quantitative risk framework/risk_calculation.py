"""
Improved Seebens Three-Layer Bioinvasion Risk Assessment
=========================================================
Computes P_alien × P_intro × P_estab* for each voyage and source port.

Inputs:
  data/observed_voyages_cleaned.csv       – 2,432 observed voyages
  data/track2CVAE_synthetic_voyages.csv   – 12,000 synthetic voyages
                                            (Layer 1 & 2 already passed)
Outputs:
  data/risk_observed.csv    – per-voyage risk for observed data
  data/risk_synthetic.csv   – per-voyage risk for synthetic data
"""

import ast
import warnings
import numpy as np
import pandas as pd
from math import radians, sin, cos, sqrt, atan2

warnings.filterwarnings("ignore")

import os
BASE    = os.path.dirname(os.path.abspath(__file__))
DATA    = os.path.join(BASE, "..", "data")
OUTPUT  = os.path.join(BASE, "..", "output")
DWT_DIR = os.path.join(BASE, "dwt_log_fit", "data")
os.makedirs(OUTPUT, exist_ok=True)

#Risk parameters (Appendix B, techdoc)
BETA      = 8        # P_alien logistic slope
GAMMA     = 1000.0   # P_alien distance threshold (km)
MU        = 0.02     # organism die-off rate (day⁻¹)
LAMBDA_BW = 0.002    # ballast-water concentration constant (m⁻³)
RHO       = 1.0      # treatment efficiency (1.0 = no treatment)
Z         = 0.95     # biotic exchange ratio
ALPHA     = 1.5e-4   # baseline establishment probability
ESR_INTRA = 0.99     # same-province ESR value (diagonal is 0 in matrix; 0.99 > max off-diagonal 0.987)
CIRCUITY  = 1.3      # sea-route / great-circle approximation factor

# NZ port → province ID
NZ_PORT_PROV = {
    "AUCKLAND":      53,
    "TAURANGA":      53,
    "GISBORNE":      53,
    "NAPIER":        54,
    "NELSON":        54,
    "NEW PLYMOUTH":  54,
    "PORT CHALMERS": 54,
    "LYTTELTON":     54,
    "TIMARU":        54,
    "WELLINGTON":    54,
}

# 1. LOAD DATA
print("Loading data …")

obs = pd.read_csv(os.path.join(DATA, "observed_voyages_cleaned.csv"))
syn = pd.read_csv(os.path.join(DATA, "track2_synthetic_voyages.csv"))

esr_df = pd.read_csv(os.path.join(DATA, "ecoprov_envdist_scaled.csv"), index_col=0)
esr_df.index   = esr_df.index.astype(int)
esr_df.columns = esr_df.columns.astype(int)

reg = pd.read_csv(os.path.join(DWT_DIR, "log_regression_results.csv"))

merged = pd.read_csv(
    os.path.join(DATA, "merged_vessel_events_pipeline_ecoprovince.csv"),
    encoding="latin-1",
    usecols=["Port name", "Latitude", "Longitude", "PROVINCE_ID", "PROVINCE_NAME"],
)


# 2. BUILD LOOKUP TABLES
print("Building lookup tables …")

def parse_list(s):
    try:
        return ast.literal_eval(str(s))
    except Exception:
        return []

# Province name → ID
prov_name_to_id = (
    merged[["PROVINCE_NAME", "PROVINCE_ID"]]
    .dropna()
    .drop_duplicates()
    .set_index("PROVINCE_NAME")["PROVINCE_ID"]
    .astype(int)
    .to_dict()
)

# Port → (lat, lon)
port_coords: dict = {}
for _, r in merged.dropna(subset=["Latitude", "Longitude"]).iterrows():
    p = r["Port name"]
    if p not in port_coords:
        port_coords[p] = (float(r["Latitude"]), float(r["Longitude"]))

for _, row in obs.iterrows():
    ports = parse_list(row["ports"])
    lats  = parse_list(str(row["latitudes"]))
    lons  = parse_list(str(row["longitudes"]))
    for p, la, lo in zip(ports, lats, lons):
        if p not in port_coords:
            try:
                port_coords[p] = (float(la), float(lo))
            except (TypeError, ValueError):
                pass

# Hạ Long – encoding-corrupted in synthetic data
port_coords["HÃ\x83Â¡Ã\x82Âº LONG"] = (20.8702, 107.0898)

# Port → province ID
port_prov_id: dict = {}
for _, r in merged.dropna(subset=["PROVINCE_ID"]).iterrows():
    p = r["Port name"]
    if p not in port_prov_id:
        port_prov_id[p] = int(r["PROVINCE_ID"])

for _, row in obs.iterrows():
    ports = parse_list(row["ports"])
    ecos  = parse_list(row["ecoprovinces"])
    for p, eco in zip(ports, ecos):
        if p not in port_prov_id and eco and eco in prov_name_to_id:
            port_prov_id[p] = prov_name_to_id[eco]

port_prov_id.update(NZ_PORT_PROV)

# Ballast-water regression model: vessel_type → (intercept, slope)
# Non-significant regressions (p ≥ 0.05) fall back to the mean of significant types
# only, to avoid using unreliable regression parameters (e.g. Reefer: p=0.76, R²=0.01).
_sig_reg = reg[reg["p_value"] < 0.05]
_fb_intercept = float(_sig_reg["intercept"].mean())
_fb_slope     = float(_sig_reg["slope"].mean())
reg_model: dict = {}
for _, r in reg.iterrows():
    if r["p_value"] < 0.05:
        reg_model[r["group"]] = (float(r["intercept"]), float(r["slope"]))
    else:
        reg_model[r["group"]] = (_fb_intercept, _fb_slope)
reg_model["_default"] = (_fb_intercept, _fb_slope)

print(f"  port_coords  : {len(port_coords):,} ports")
print(f"  port_prov_id : {len(port_prov_id):,} ports with province")


# 3. HELPER FUNCTIONS

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    lat1_r, lat2_r = radians(lat1), radians(lat2)
    dlat = lat2_r - lat1_r
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(lat1_r) * cos(lat2_r) * sin(dlon / 2) ** 2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))

def get_gc_dist_km(port_a: str, port_b: str) -> float | None:
    """Great-circle distance (km) – used for P_alien (techdoc §8.2: 大圆距离)."""
    ca = port_coords.get(port_a)
    cb = port_coords.get(port_b)
    if ca is None or cb is None:
        return None
    return haversine_km(ca[0], ca[1], cb[0], cb[1])

def predict_wr(vessel_type: str, dwt: float) -> float:
    vt = vessel_type if vessel_type in reg_model else "_default"
    a, b = reg_model[vt]
    return float(np.exp(a + b * np.log(max(dwt, 1.0))))

def compute_delta_t_hours(i: int, n: int, transits: list, dwells: list) -> float:
    return sum(transits[i + 1: n + 1]) + sum(dwells[i + 1: n])


# 4. THREE-FILTER RISK FUNCTIONS

def p_alien(dist_km: float) -> float:
    # β=8 is calibrated for d in Mm (thousands of km); GAMMA=1000 km = 1 Mm
    return float(1.0 / (1.0 + np.exp(-BETA * (dist_km - GAMMA) / 1000.0)))


def p_intro(delta_t_hours: float, delta_r: int, dwt: float, vessel_type: str) -> float:
    delta_t_days = delta_t_hours / 24.0
    W_r = predict_wr(vessel_type, dwt)
    V_r = 0.3 * dwt
    if V_r <= 0 or W_r <= 0:
        return 0.0
    exchange_ratio = min(Z * W_r / V_r, 0.9999)
    B_r  = Z * W_r * ((1.0 - exchange_ratio) ** delta_r)
    prob = RHO * (1.0 - np.exp(-LAMBDA_BW * B_r)) * np.exp(-MU * delta_t_days)
    return float(np.clip(prob, 0.0, 1.0))


def p_estab(src_prov_id, nz_prov_id) -> float:
    if src_prov_id is None or nz_prov_id is None:
        return 0.0
    if src_prov_id == nz_prov_id:
        esr_val = ESR_INTRA
    elif src_prov_id in esr_df.index and nz_prov_id in esr_df.columns:
        esr_val = float(esr_df.loc[src_prov_id, nz_prov_id])
    else:
        return 0.0
    return float(ALPHA * esr_val)


def compute_voyage_risk(
    ports: list, transits: list, dwells: list,
    n_ports: int, dwt: float, vessel_type: str,
) -> dict:
    n        = n_ports - 1
    nz_port  = ports[n]
    nz_prov  = port_prov_id.get(nz_port)

    pa_list, pi_list, pe_list, pr_list = [], [], [], []

    for i in range(n):
        src_port = ports[i]

        d_km = get_gc_dist_km(src_port, nz_port)
        pa   = p_alien(d_km) if d_km is not None else 0.5

        dt_h    = compute_delta_t_hours(i, n, transits, dwells)
        delta_r = n - i - 1
        pi      = p_intro(dt_h, delta_r, dwt, vessel_type)

        pe = p_estab(port_prov_id.get(src_port), nz_prov)

        pr = pa * pi * pe
        pa_list.append(round(pa, 6))
        pi_list.append(round(pi, 8))
        pe_list.append(round(pe, 8))
        pr_list.append(round(pr, 10))

    p_inv = float(1.0 - np.prod([1.0 - p for p in pr_list]))

    return {
        "p_alien_per_port": pa_list,
        "p_intro_per_port": pi_list,
        "p_estab_per_port": pe_list,
        "pr_inv_per_port":  pr_list,
        "p_invasion":       round(p_inv, 10),
    }


# 5. PROCESS OBSERVED DATA
print("\nProcessing observed voyages …")

obs_results = []
obs_skipped = 0

for idx, row in obs.iterrows():
    ports    = parse_list(row["ports"])
    transits = parse_list(str(row["transits_hours"]))
    dwells   = parse_list(str(row["dwells_hours"]))
    n_ports  = int(row["n_ports"])
    dwt      = float(row["DWT"])
    vtype    = str(row["nbic_type_group"])
    nz_port  = str(row["nz_dest_port"])

    if n_ports < 2 or len(ports) != n_ports:
        obs_skipped += 1
        continue

    risk = compute_voyage_risk(ports, transits, dwells, n_ports, dwt, vtype)

    obs_results.append({
        "voyage_id":    row["voyage_id"],
        "vessel_type":  vtype,
        "dwt":          dwt,
        "n_ports":      n_ports,
        "nz_dest_port": nz_port,
        "cluster_k":    row.get("cluster_k", np.nan),
        **risk,
    })

obs_out = pd.DataFrame(obs_results)
print(f"  Processed : {len(obs_out):,}   skipped: {obs_skipped}")
print(f"  p_invasion  mean={obs_out['p_invasion'].mean():.4e}  "
      f"median={obs_out['p_invasion'].median():.4e}  "
      f"max={obs_out['p_invasion'].max():.4e}")


# 6. PROCESS SYNTHETIC DATA
print("\nProcessing synthetic voyages …")

syn_results = []
syn_skipped = 0

for idx, row in syn.iterrows():
    ports    = parse_list(row["ports"])
    transits = parse_list(str(row["transits_hours"]))
    dwells   = parse_list(str(row["dwells_hours"]))
    n_ports  = int(row["n_ports"])
    dwt      = float(row["dwt"])
    vtype    = str(row["vessel_type"])
    nz_port  = str(row["nz_destination"])

    if n_ports < 2 or len(ports) != n_ports:
        syn_skipped += 1
        continue

    risk = compute_voyage_risk(ports, transits, dwells, n_ports, dwt, vtype)

    syn_results.append({
        "voyage_id":    idx,
        "vessel_type":  vtype,
        "dwt":          dwt,
        "n_ports":      n_ports,
        "nz_dest_port": nz_port,
        "cluster_k":    int(row["cluster_k"]),
        **risk,
    })

syn_out = pd.DataFrame(syn_results)
print(f"  Processed : {len(syn_out):,}   skipped: {syn_skipped}")
print(f"  p_invasion  mean={syn_out['p_invasion'].mean():.4e}  "
      f"median={syn_out['p_invasion'].median():.4e}  "
      f"max={syn_out['p_invasion'].max():.4e}")


# 7. SAVE RESULTS
print("\nSaving results …")

obs_path = os.path.join(OUTPUT, "risk_observed.csv")
syn_path = os.path.join(OUTPUT, "track2_risk_synthetic.csv")

obs_out.to_csv(obs_path, index=False)
syn_out.to_csv(syn_path, index=False)

print(f"  {obs_path}  ({len(obs_out):,} rows)")
print(f"  {syn_path}  ({len(syn_out):,} rows)")
print("\nDone.")
