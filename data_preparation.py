"""
Data preparation: filtering, format conversion and descriptive statistics.
Full pipeline: Steps 1-4 + NBIC sampling.
"""

import ast
from collections import Counter

import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

BASE   = Path(__file__).resolve().parent
DATA   = BASE / "data"
OUTPUT = BASE / "output"
OUTPUT.mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────
# Vessel type mapping  (merged.csv Type → NBIC category)
# ─────────────────────────────────────────────────────────────
TYPE_MAP = {
    "Bulk Carrier":               "Bulker",
    "Ore/Bulk Carrier":           "Bulker",
    "Wood-Chip Carrier":          "Bulker",
    "Container Ship":             "Container",
    "General Cargo":              "General Cargo",
    "Cargo":                      "General Cargo",
    "Cargo - Hazard A":           "General Cargo",
    "Cargo - Hazard C":           "General Cargo",
    "Cargo - Hazard D":           "General Cargo",
    "Cargo - Fish carrier":       "General Cargo",
    "Heavy Load Carrier":         "General Cargo",
    "Livestock Carrier":          "General Cargo",
    "Tanker":                     "Tanker",
    "Tanker - Hazard A":          "Tanker",
    "Tanker - Hazard B":          "Tanker",
    "Tanker - Hazard C":          "Tanker",
    "Tanker - Hazard D":          "Tanker",
    "Oil/ Chemical Tanker":       "Tanker",
    "Crude Oil Tanker":           "Tanker",
    "Vehicle Carrier":            "RoRo",
    "Ro-Ro Cargo":                "RoRo",
    "Reefer":                     "Reefer",
    "Passenger Ship":             "Passenger",
    "High Speed Passenger Craft": "Passenger",
    "Research Vessel":            "Other",
    "Tugboat":                    "Other",
    "Offshore Supply Ship":       "Other",
    "Other Type of Ship":         "Other",
}


# ════════════════════════════════════════════════════════════
# Load raw data
# ════════════════════════════════════════════════════════════
print("=" * 60)
print("Loading merged_vessel_events dataset")
print("=" * 60)

df = pd.read_csv(DATA / "merged_vessel_events_pipeline_ecoprovince.csv",
                 parse_dates=["Event start date & time", "Event end date & time"])

print(f"Raw row count    : {len(df):,}")
print(f"Unique voyage_id : {df['voyage_id'].nunique():,}")
print(f"Unique vessels (MMSI): {df['MMSI'].nunique():,}")


# ════════════════════════════════════════════════════════════
# STEP 1: Filter valid voyages (n_ports >= 5)
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 1: Filter valid voyages")
print("=" * 60)

# 1a. Count ports per voyage
port_counts = df.groupby("voyage_id").size().rename("n_ports")
df = df.join(port_counts, on="voyage_id")

# 1b. Keep voyages with n_ports >= 5
df_valid = df[df["n_ports"] >= 5].copy()
print(f"n_ports >= 5 → retained rows: {len(df_valid):,},  voyages: {df_valid['voyage_id'].nunique():,}")

# 1c. Verify that the final event (latest timestamp) is in New Zealand
def check_nz_terminal(group):
    grp = group.sort_values("Event start date & time")
    return grp.iloc[-1]["Port territory"] == "New Zealand"

nz_ok = df_valid.groupby("voyage_id").apply(check_nz_terminal)
invalid_voyages = nz_ok[~nz_ok].index
print(f"Voyages failing NZ-terminal check: {len(invalid_voyages)}")
df_valid = df_valid[~df_valid["voyage_id"].isin(invalid_voyages)]

# 1d. Check for duplicate intermediate ports
def has_duplicate_middle_ports(group):
    grp = group.sort_values("Event start date & time")
    middle = grp["Port name"].iloc[1:-1].tolist()
    return len(middle) != len(set(middle))

dup_flag = df_valid.groupby("voyage_id").apply(has_duplicate_middle_ports)
dup_voyages = dup_flag[dup_flag].index
print(f"Voyages with duplicate intermediate ports: {len(dup_voyages)}")
df_valid = df_valid[~df_valid["voyage_id"].isin(dup_voyages)]

# 1e. Timestamp completeness (already parsed; confirm no NaT)
ts_null = df_valid["Event start date & time"].isna().sum()
print(f"Rows with missing timestamps: {ts_null}")

n_voyages_final = df_valid["voyage_id"].nunique()
n_vessels_final = df_valid["MMSI"].nunique()
n_imo_final     = df_valid["IMO"].nunique()
print(f"\n✓ Step 1 done → valid voyages: {n_voyages_final:,},  unique vessels: {n_vessels_final:,}  (IMO: {n_imo_final:,})")

# n_ports distribution
nports_dist = df_valid.drop_duplicates("voyage_id")["n_ports"].value_counts().sort_index()
print("\nn_ports distribution (valid voyages):")
for v, c in nports_dist.items():
    print(f"  {v} ports: {c:,}  ({c/n_voyages_final*100:.1f}%)")


# ════════════════════════════════════════════════════════════
# STEP 2: Convert to sequential data format
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 2: Convert to sequential format")
print("=" * 60)

# Apply vessel type mapping
df_valid["nbic_type_group"] = df_valid["Type"].map(TYPE_MAP).fillna("Other")

# Report types mapped to Other
unmapped = df_valid[df_valid["nbic_type_group"] == "Other"]["Type"].unique()
print(f"Raw types mapped to 'Other': {unmapped}")

def aggregate_voyage(group):
    grp  = group.sort_values("Event start date & time")
    row  = grp.iloc[0]
    last = grp.iloc[-1]
    return pd.Series({
        "mmsi":                 row["MMSI"],
        "imo":                  row["IMO"],
        "vessel_name":          row["Vessel name"],
        "vessel_type_raw":      row["Type"],
        "nbic_type_group":      row["nbic_type_group"],
        "n_ports":              len(grp),
        "ports":                grp["Port name"].tolist(),
        "port_territories":     grp["Port territory"].tolist(),
        "latitudes":            grp["Latitude"].tolist(),
        "longitudes":           grp["Longitude"].tolist(),
        "ecoprovinces":         grp["PROVINCE_NAME"].tolist(),
        "dwells_hours":         (grp["Event duration (seconds)"] / 3600).tolist(),
        "transits_hours":       grp["voyage_duration"].tolist(),
        "voyage_start_dt":      grp["Event start date & time"].iloc[0],
        "voyage_end_dt":        last["Event start date & time"],
        "total_span_days":      (last["Event start date & time"] -
                                 grp["Event start date & time"].iloc[0]).total_seconds() / 86400,
        "env_similarity_risks": grp["Environmental_Similarity_Risk"].tolist(),
        "nz_dest_port":         last["Port name"],
    })

seq_df = df_valid.groupby("voyage_id").apply(aggregate_voyage).reset_index()
print(f"Sequential format output: {len(seq_df):,} rows (one row per voyage)")
seq_df.to_csv(OUTPUT / "observed_voyages_sequential.csv", index=False)
print("✓ Saved → output/observed_voyages_sequential.csv")


# ════════════════════════════════════════════════════════════
# STEP 3: Descriptive statistics
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 3: Descriptive statistics")
print("=" * 60)

# ── 3.1 Voyage-level statistics ───────────────────────────
print("\n[3.1 Voyage-level statistics]")

# total_span_days
tsd = seq_df["total_span_days"]
print(f"\ntotal_span_days (total voyage duration, days):")
print(f"  mean: {tsd.mean():.2f}   median: {tsd.median():.2f}   std: {tsd.std():.2f}")
print(f"  p5={tsd.quantile(0.05):.2f}   p25={tsd.quantile(0.25):.2f}   "
      f"p75={tsd.quantile(0.75):.2f}   p95={tsd.quantile(0.95):.2f}")
print(f"  max: {tsd.max():.2f}   anomalously long voyages (>60 days): {(tsd > 60).sum()}")

# avg_transit_hours (excluding the first port's zero value)
def avg_transit(row):
    t = row["transits_hours"]
    if isinstance(t, str):
        t = ast.literal_eval(t)
    nonzero = [x for x in t[1:] if x > 0]
    return np.mean(nonzero) if nonzero else np.nan

seq_df["avg_transit_hours"] = seq_df.apply(avg_transit, axis=1)
ath = seq_df["avg_transit_hours"].dropna()
print(f"\navg_transit_hours (mean transit time per leg, hours):")
print(f"  mean: {ath.mean():.1f}   median: {ath.median():.1f}   std: {ath.std():.1f}")
print(f"  p5={ath.quantile(0.05):.1f}   p25={ath.quantile(0.25):.1f}   "
      f"p75={ath.quantile(0.75):.1f}   p95={ath.quantile(0.95):.1f}")

# avg_dwell_hours
def avg_dwell(row):
    d = row["dwells_hours"]
    if isinstance(d, str):
        d = ast.literal_eval(d)
    return np.mean(d)

seq_df["avg_dwell_hours"] = seq_df.apply(avg_dwell, axis=1)
adh = seq_df["avg_dwell_hours"]
print(f"\navg_dwell_hours (mean port dwell time, hours):")
print(f"  mean: {adh.mean():.1f}   median: {adh.median():.1f}   std: {adh.std():.1f}")
print(f"  p5={adh.quantile(0.05):.1f}   p25={adh.quantile(0.25):.1f}   "
      f"p75={adh.quantile(0.75):.1f}   p95={adh.quantile(0.95):.1f}")
print(f"  voyages with dwell < 6 h (anomalously short): {(adh < 6).sum()}")
print(f"  voyages with dwell > 96 h (anomalously long): {(adh > 96).sum()}")

# n_ports distribution
print(f"\nn_ports distribution:")
for v, c in nports_dist.items():
    print(f"  {v} ports: {c:,}  ({c/n_voyages_final*100:.1f}%)")


# ── 3.2 Vessel type mapping validation ────────────────────
print("\n[3.2 Vessel type mapping validation (merged.csv Type → NBIC)]")
type_cross = df_valid.drop_duplicates("voyage_id")[["Type", "nbic_type_group"]]
print(type_cross.groupby(["nbic_type_group", "Type"]).size().to_string())


# ── 3.3 Vessel-type-level statistics ──────────────────────
print("\n[3.3 Vessel-type statistics (by nbic_type_group)]")

type_voyage_cnt = seq_df["nbic_type_group"].value_counts()
type_vessel_cnt = seq_df.groupby("nbic_type_group")["imo"].nunique()

print(f"\n{'Vessel type':<20} {'Voyages':>8} {'Share %':>8} {'Unique IMO':>11}")
print("-" * 50)
for g in type_voyage_cnt.index:
    n_v = type_voyage_cnt[g]
    n_i = type_vessel_cnt.get(g, 0)
    print(f"{g:<20} {n_v:>8,} {n_v/n_voyages_final*100:>8.1f} {n_i:>11,}")


# ── 3.3b Time-parameter distribution by vessel type ───────
print("\n[3.3b Time-parameter distribution by vessel type]")
print(f"\n{'Vessel type':<20} {'AvgTransit mean':>16} {'AvgTransit median':>18} "
      f"{'AvgDwell mean':>14} {'AvgDwell median':>16} {'Span mean (days)':>17}")
print("-" * 95)
for g in type_voyage_cnt.index:
    sub   = seq_df[seq_df["nbic_type_group"] == g]
    at_m  = sub["avg_transit_hours"].mean()
    at_md = sub["avg_transit_hours"].median()
    ad_m  = sub["avg_dwell_hours"].mean()
    ad_md = sub["avg_dwell_hours"].median()
    sp_m  = sub["total_span_days"].mean()
    print(f"{g:<20} {at_m:>16.1f} {at_md:>18.1f} {ad_m:>14.1f} {ad_md:>16.1f} {sp_m:>17.1f}")


# ── 3.4 Port sequence analysis ────────────────────────────
print("\n[3.4 Port sequence analysis]")

# Origin territory distribution
def get_origin_territory(row):
    pt = row["port_territories"]
    if isinstance(pt, str):
        pt = ast.literal_eval(pt)
    return pt[0]

seq_df["origin_territory"] = seq_df.apply(get_origin_territory, axis=1)
ot_dist = seq_df["origin_territory"].value_counts().head(15)
print(f"\nOrigin territory distribution (top 15):")
for t, c in ot_dist.items():
    print(f"  {t:<30} {c:>5,}  ({c/n_voyages_final*100:.1f}%)")

# NZ destination port distribution
nz_dest = seq_df["nz_dest_port"].value_counts()
print(f"\nNZ destination port distribution:")
for p, c in nz_dest.items():
    print(f"  {p:<30} {c:>5,}  ({c/n_voyages_final*100:.1f}%)")

# Ecoprovince coverage (source ports only, excluding NZ terminal)
def get_source_ecoprovinces(row):
    ep = row["ecoprovinces"]
    if isinstance(ep, str):
        ep = ast.literal_eval(ep)
    return ep[:-1]

all_source_ep = []
for _, row in seq_df.iterrows():
    all_source_ep.extend(get_source_ecoprovinces(row))
unique_ep = {ep for ep in all_source_ep
             if isinstance(ep, str) and ep not in ("None", "nan", "")}
print(f"\nUnique MEOW ecoprovinces in source ports: {len(unique_ep)}")
print(f"Ecoprovince list:")
for ep in sorted(unique_ep):
    print(f"  {ep}")

# High-frequency adjacent port pairs (top 20)
port_pairs = []
for _, row in seq_df.iterrows():
    ports = row["ports"]
    if isinstance(ports, str):
        ports = ast.literal_eval(ports)
    for i in range(len(ports) - 1):
        port_pairs.append(f"{ports[i]} → {ports[i+1]}")

pair_cnt = Counter(port_pairs)
print(f"\nHigh-frequency adjacent port pairs (top 20):")
for pair, cnt in pair_cnt.most_common(20):
    print(f"  {pair:<55} {cnt:>4}")


# ════════════════════════════════════════════════════════════
# STEP 4: Generate Equasis scraping list
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 4: Generate Equasis scraping list")
print("=" * 60)

# List A: all vessels in valid voyages
list_a = seq_df.groupby("imo").agg(
    mmsi=("mmsi", "first"),
    vessel_name=("vessel_name", "first"),
    vessel_type_raw=("vessel_type_raw", "first"),
    nbic_type_group=("nbic_type_group", "first"),
    n_voyages=("voyage_id", "count")
).reset_index()

list_a = list_a[list_a["imo"] > 0]   # drop records with IMO = 0
list_a.to_csv(OUTPUT / "equasis_list_A_voyage_vessels.csv", index=False)
print(f"\nList A: {len(list_a):,} unique IMO vessels → output/equasis_list_A_voyage_vessels.csv")


# ════════════════════════════════════════════════════════════
# NBIC data processing: sample vessels from arrivals.csv for List B
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("NBIC data processing: sampling from arrivals.csv")
print("=" * 60)

nbic = pd.read_csv(DATA / "merged_vessel_events_pipeline_ecoprovince.csv", parse_dates=["Arrival_Date"])
print(f"NBIC raw row count: {len(nbic):,}")

# Keep only records with a valid IMO number
nbic_imo = nbic[nbic["ID_Type"] == "IMO number"].copy()
nbic_imo["IMO_Number"] = pd.to_numeric(nbic_imo["IMO_Number"], errors="coerce")
nbic_imo = nbic_imo[nbic_imo["IMO_Number"].notna() & (nbic_imo["IMO_Number"] > 0)]
print(f"Records with valid IMO: {len(nbic_imo):,}")

# Keep only overseas transits (actual ballast water discharge events)
nbic_overseas = nbic_imo[nbic_imo["Transit_Type"] == "Overseas"].copy()
print(f"Overseas transit records: {len(nbic_overseas):,}")

# Keep only records with actual discharge volume (sum_Overseas_MT > 0)
nbic_with_bw = nbic_overseas[nbic_overseas["sum_Overseas_MT"] > 0].copy()
print(f"Records with ballast water discharge (sum_Overseas_MT > 0): {len(nbic_with_bw):,}")

# Deduplicate: one record per vessel (most recent arrival)
nbic_dedup = (nbic_with_bw
              .sort_values("Arrival_Date", ascending=False)
              .drop_duplicates("IMO_Number")
              .copy())
print(f"Unique vessels after deduplication: {len(nbic_dedup):,}")

# Summary by NBIC Vessel_Type
print(f"\nNBIC valid sample pool (by Vessel_Type):")
vt_cnt = nbic_with_bw["Vessel_Type"].value_counts()
for vt, cnt in vt_cnt.items():
    print(f"  {vt:<20} {cnt:>6,} records")

# NBIC category → dataset category: stratified sampling targets
NBIC_TARGET = {
    "Bulker":        50,
    "Container":     50,
    "General Cargo": 40,
    "Tanker":        40,
    "RoRo":          20,
    "Reefer":        15,
    "Passenger":     10,
    "Other":         10,
}

# Stratified sampling by GT (used as DWT proxy), three size tiers
sample_list_b_parts = []

print(f"\nStratified sampling results:")
print(f"{'NBIC type':<20} {'Target':>7} {'Pool':>6} {'Small':>6} {'Medium':>7} {'Large':>6} {'Sampled':>8}")
print("-" * 65)

for vtype, target in NBIC_TARGET.items():
    pool = nbic_with_bw[nbic_with_bw["Vessel_Type"] == vtype].copy()
    pool["GT"] = pd.to_numeric(pool["GT"], errors="coerce")
    pool = pool[pool["GT"].notna() & (pool["GT"] > 0)]

    # One record per vessel (most recent arrival)
    pool = (pool.sort_values("Arrival_Date", ascending=False)
                .drop_duplicates("IMO_Number"))

    if len(pool) == 0:
        print(f"{vtype:<20} {target:>7} {'0':>6} {'—':>6} {'—':>7} {'—':>6} {'0':>8}")
        continue

    # GT terciles
    p33 = pool["GT"].quantile(0.33)
    p67 = pool["GT"].quantile(0.67)
    pool["gt_tier"] = pd.cut(pool["GT"],
                              bins=[-np.inf, p33, p67, np.inf],
                              labels=["Small", "Medium", "Large"])

    per_tier = max(1, target // 3)
    sampled, tier_counts = [], {}
    for tier in ["Small", "Medium", "Large"]:
        tier_pool = pool[pool["gt_tier"] == tier]
        n_take = min(per_tier, len(tier_pool))
        if n_take > 0:
            sampled.append(tier_pool.sample(n=n_take, random_state=42))
        tier_counts[tier] = n_take

    if sampled:
        sampled_df = pd.concat(sampled)
        sampled_df["nbic_type_group"] = vtype
        sampled_df["in_list_a"] = sampled_df["IMO_Number"].isin(list_a["imo"]).astype(bool)
        sample_list_b_parts.append(sampled_df)
        print(f"{vtype:<20} {target:>7} {len(pool):>6} "
              f"{tier_counts.get('Small',0):>6} {tier_counts.get('Medium',0):>7} "
              f"{tier_counts.get('Large',0):>6} {len(sampled_df):>8}")

# Compile List B
list_b_raw = pd.concat(sample_list_b_parts, ignore_index=True)

list_b = list_b_raw[[
    "IMO_Number", "Vessel_Name", "Vessel_Type", "GT",
    "nbic_type_group", "gt_tier", "in_list_a",
    "sum_Overseas_MT", "Arrival_Date"
]].rename(columns={
    "IMO_Number":    "imo",
    "Vessel_Name":   "vessel_name",
    "Vessel_Type":   "vessel_type_nbic",
    "GT":            "gt_proxy",
    "gt_tier":       "gt_size_tier",
    "sum_Overseas_MT": "sample_bw_mt",
    "Arrival_Date":  "sample_arrival_date"
})

# Save NBIC paired records (used for regression)
nbic_sample_records = nbic_with_bw[
    nbic_with_bw["IMO_Number"].isin(list_b["imo"])
].copy()
nbic_sample_records.to_csv(OUTPUT / "nbic_ballast_sample.csv", index=False)

list_b.to_csv(OUTPUT / "equasis_list_B_nbic_sample.csv", index=False)
overlap = list_b["in_list_a"].sum()
print(f"\nList B total: {len(list_b):,} vessels")
print(f"  Overlap with List A: {overlap:,}  (flagged in_list_a=True)")
print(f"✓ Saved → output/equasis_list_B_nbic_sample.csv")
print(f"✓ Saved → output/nbic_ballast_sample.csv  ({len(nbic_sample_records):,} ballast water records)")


# ════════════════════════════════════════════════════════════
# Export descriptive statistics summary tables (CSV)
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Exporting descriptive statistics summary")
print("=" * 60)

# Voyage-level summary
voyage_summary = pd.DataFrame({
    "metric": [
        "Total valid voyages",
        "Unique vessels (MMSI)",
        "Unique vessels (IMO)",
        "n_ports=5 count", "n_ports=5 share %",
        "n_ports=6 count", "n_ports=6 share %",
        "n_ports=7 count", "n_ports=7 share %",
        "total_span_days mean",
        "total_span_days median",
        "total_span_days std",
        "total_span_days p5",
        "total_span_days p95",
        "Anomalously long voyages (>60 days)",
        "avg_transit_hours mean",
        "avg_transit_hours median",
        "avg_transit_hours p5",
        "avg_transit_hours p95",
        "avg_dwell_hours mean",
        "avg_dwell_hours median",
        "avg_dwell_hours p5",
        "avg_dwell_hours p95",
        "Voyages with avg_dwell < 6 h",
        "Voyages with avg_dwell > 96 h",
    ],
    "value": [
        n_voyages_final,
        n_vessels_final,
        n_imo_final,
        nports_dist.get(5, 0), f"{nports_dist.get(5, 0)/n_voyages_final*100:.1f}",
        nports_dist.get(6, 0), f"{nports_dist.get(6, 0)/n_voyages_final*100:.1f}",
        nports_dist.get(7, 0), f"{nports_dist.get(7, 0)/n_voyages_final*100:.1f}",
        f"{tsd.mean():.2f}",
        f"{tsd.median():.2f}",
        f"{tsd.std():.2f}",
        f"{tsd.quantile(0.05):.2f}",
        f"{tsd.quantile(0.95):.2f}",
        int((tsd > 60).sum()),
        f"{ath.mean():.1f}",
        f"{ath.median():.1f}",
        f"{ath.quantile(0.05):.1f}",
        f"{ath.quantile(0.95):.1f}",
        f"{adh.mean():.1f}",
        f"{adh.median():.1f}",
        f"{adh.quantile(0.05):.1f}",
        f"{adh.quantile(0.95):.1f}",
        int((adh < 6).sum()),
        int((adh > 96).sum()),
    ]
})
voyage_summary.to_csv(OUTPUT / "descriptive_stats_voyage.csv", index=False)

# Vessel-type-level summary
type_stats_rows = []
for g in type_voyage_cnt.index:
    sub = seq_df[seq_df["nbic_type_group"] == g]
    at  = sub["avg_transit_hours"]
    ad  = sub["avg_dwell_hours"]
    sp  = sub["total_span_days"]
    type_stats_rows.append({
        "nbic_type_group":      g,
        "voyage_count":         len(sub),
        "voyage_pct":           round(len(sub)/n_voyages_final*100, 1),
        "unique_imo":           sub["imo"].nunique(),
        "avg_transit_mean_h":   round(at.mean(), 1),
        "avg_transit_median_h": round(at.median(), 1),
        "avg_transit_std_h":    round(at.std(), 1),
        "avg_dwell_mean_h":     round(ad.mean(), 1),
        "avg_dwell_median_h":   round(ad.median(), 1),
        "avg_dwell_std_h":      round(ad.std(), 1),
        "span_mean_days":       round(sp.mean(), 1),
        "span_median_days":     round(sp.median(), 1),
        "span_std_days":        round(sp.std(), 1),
    })
type_stats_df = pd.DataFrame(type_stats_rows)
type_stats_df.to_csv(OUTPUT / "descriptive_stats_by_type.csv", index=False)

# Origin territory distribution
(ot_dist.reset_index()
 .rename(columns={"index": "origin_territory", "count": "voyage_count"})
 .to_csv(OUTPUT / "origin_territory_dist.csv", index=False))

# NZ destination port distribution
(nz_dest.reset_index()
 .rename(columns={"index": "nz_dest_port", "count": "voyage_count"})
 .to_csv(OUTPUT / "nz_dest_port_dist.csv", index=False))

# High-frequency port pairs
pair_df = pd.DataFrame(pair_cnt.most_common(50), columns=["port_pair", "count"])
pair_df.to_csv(OUTPUT / "top_port_pairs.csv", index=False)

# Ecoprovince coverage
ep_df = pd.DataFrame(sorted(unique_ep), columns=["source_ecoprovince"])
ep_df.to_csv(OUTPUT / "source_ecoprovinces.csv", index=False)

print(f"\n✓ All descriptive statistics saved to output/")
print(f"   descriptive_stats_voyage.csv")
print(f"   descriptive_stats_by_type.csv")
print(f"   origin_territory_dist.csv")
print(f"   nz_dest_port_dist.csv")
print(f"   top_port_pairs.csv")
print(f"   source_ecoprovinces.csv")

print("\n" + "=" * 60)
print("All steps complete")
print("=" * 60)
