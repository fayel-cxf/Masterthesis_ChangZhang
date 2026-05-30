import ast
import numpy as np
import pandas as pd
from math import radians, cos, sin, asin, sqrt
from pathlib import Path
from sklearn.preprocessing import StandardScaler

# ------------------------------------------------------------------
# Path Configuration
# ------------------------------------------------------------------
BASE_DIR   = Path(__file__).resolve().parent.parent.parent
INPUT_PATH = BASE_DIR / "data/clustering/observed_voyages_cleaned.csv"
OUTPUT_PATH = BASE_DIR / "data/clustering/cluster.m1.prep.csv"

# ------------------------------------------------------------------
# Read Raw Data
# ------------------------------------------------------------------
df = pd.read_csv(INPUT_PATH)
print(f"[Read] {df.shape[0]} rows × {df.shape[1]} columns")

# ------------------------------------------------------------------
# Step 1: Parse List-type Columns
# ------------------------------------------------------------------
def parse_list_col(series):
    return series.apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

df["latitudes"]      = parse_list_col(df["latitudes"])
df["longitudes"]     = parse_list_col(df["longitudes"])
df["dwells_hours"]   = parse_list_col(df["dwells_hours"])
df["transits_hours"] = parse_list_col(df["transits_hours"])
print("[Step 1] List-type columns parsing completed")

# ------------------------------------------------------------------
# Step 2: Calculate Derived Features
# ------------------------------------------------------------------

# [2a] Temporal Features
df["avg_dwell_time"]         = df["dwells_hours"].apply(
    lambda x: np.mean(x) if len(x) > 0 else np.nan)
df["voyage_duration_total"]  = df["transits_hours"].apply(np.sum)

# [2b] Distance Features (Haversine)
def _haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    return 2 * 6371 * asin(sqrt(a))

def leg_distances(lats, lons):
    return [_haversine(lats[i], lons[i], lats[i+1], lons[i+1])
            for i in range(len(lats) - 1)]

df["max_distance_per_leg"] = df.apply(
    lambda r: max(leg_distances(r["latitudes"], r["longitudes"]))
              if len(r["latitudes"]) > 1 else np.nan, axis=1)
df["total_distance"] = df.apply(
    lambda r: sum(leg_distances(r["latitudes"], r["longitudes"]))
              if len(r["latitudes"]) > 1 else np.nan, axis=1)

# [2c] Initial Geographic Location
df["origin_latitude"]  = df["latitudes"].apply(lambda x: x[0])
df["origin_longitude"] = df["longitudes"].apply(lambda x: x[0])

# [2d] Route Geographic Span
df["longitude_range"]     = df["longitudes"].apply(lambda x: max(x) - min(x))
df["hemisphere_crossing"] = df["latitudes"].apply(
    lambda x: 1.0 if (min(x) < 0 and max(x) > 0) else 0.0)

# [2e] Port Dwell Ratio
total_time = df["avg_dwell_time"] * df["n_ports"] + df["voyage_duration_total"]
df["dwell_ratio"] = (df["avg_dwell_time"] * df["n_ports"]) / total_time.replace(0, np.nan)

# [2f] Number of Territories/Countries Visited
df["n_territories"] = df["port_territories"].apply(
    lambda x: len(set(ast.literal_eval(x))) if isinstance(x, str) else np.nan)

print("[Step 2] Derived features calculation completed")

# ------------------------------------------------------------------
# Step 3: One-hot Encode Vessel Types
# ------------------------------------------------------------------
VESSEL_TYPES = ["Bulker", "Container", "General Cargo", "Other",
                "Passenger", "Reefer", "RoRo", "Tanker"]
for vt in VESSEL_TYPES:
    df[f"type_{vt}"] = (df["nbic_type_group"] == vt).astype(int)
print("[Step 3] Vessel type one-hot encoding completed")

# ------------------------------------------------------------------
# Step 4: Drop Redundant Temporal Variables
#         voyage_duration_total (r=0.821 with total_span_days)
#         Only used to calculate dwell_ratio, dropped after calculation
# ------------------------------------------------------------------
df.drop(columns=["voyage_duration_total"], inplace=True)
print("[Step 4] Redundant temporal variables dropped")

# ------------------------------------------------------------------
# Step 5: Cyclic Encoding for Longitude
# ------------------------------------------------------------------
df["lon_sin"] = np.sin(np.radians(df["origin_longitude"]))
df["lon_cos"] = np.cos(np.radians(df["origin_longitude"]))
df.drop(columns=["origin_longitude"], inplace=True)
print("[Step 5] Longitude cyclic encoding completed")

# ------------------------------------------------------------------
# Step 6: Log1p Transformation
# ------------------------------------------------------------------
for col in ["avg_dwell_time", "DWT"]:
    df[col] = np.log1p(df[col])
print("[Step 6] log1p transformation completed (avg_dwell_time, DWT)")

# ------------------------------------------------------------------
# Step 7: Combine Feature Matrix and Standardize
# ------------------------------------------------------------------
CONTINUOUS_COLS = [
    "n_ports", "total_span_days", "avg_dwell_time", "dwell_ratio",
    "DWT", "total_distance", "max_distance_per_leg",
    "origin_latitude", "lon_sin", "lon_cos",
    "longitude_range", "hemisphere_crossing", "n_territories",
]
ONEHOT_COLS = [f"type_{vt}" for vt in VESSEL_TYPES]

df_feat = df[CONTINUOUS_COLS + ONEHOT_COLS].copy()

# Step 8: Drop Rows with Missing Values
mask_valid = df_feat.notna().all(axis=1)
n_dropped = (~mask_valid).sum()
if n_dropped > 0:
    print(f"[Step 8] Warning: Dropped {n_dropped} rows containing missing values")
df_feat = df_feat[mask_valid].reset_index(drop=True)

# Standardize Continuous Features
scaler = StandardScaler()
df_feat[CONTINUOUS_COLS] = scaler.fit_transform(df_feat[CONTINUOUS_COLS])
print("[Step 7] Standardization completed")

# ------------------------------------------------------------------
# Save
# ------------------------------------------------------------------
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
df_feat.to_csv(OUTPUT_PATH, index=False)
print(f"\n[Completed] Feature matrix {df_feat.shape[0]} rows × {df_feat.shape[1]} columns")
print(f"[Output] {OUTPUT_PATH}")
print(f"Columns: {list(df_feat.columns)}")