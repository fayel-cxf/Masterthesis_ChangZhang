import os
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.gridspec import GridSpec
from pathlib import Path

import geopandas as gpd

BASE_DIR = Path(__file__).parent.parent.parent  # cluster/ -> code/ -> Project Root

os.makedirs(BASE_DIR / "graphics/cluster.pics", exist_ok=True)

# ------------------------------------------------------------------
# Path Configuration
# ------------------------------------------------------------------
INPUT_CSV          = BASE_DIR / "data/clustering/cluster.m1.labels.k5.csv"
OUTPUT_WORLD       = BASE_DIR / "graphics/cluster.pics/voyage_cluster_map_world.png"
OUTPUT_PANEL       = BASE_DIR / "graphics/cluster.pics/voyage_cluster_map_panel.png"
OUTPUT_CLUSTER_FMT = str(BASE_DIR / "graphics/cluster.pics/voyage_cluster_map_c{}.png")

# ------------------------------------------------------------------
# 0. Load world shapefile
# ------------------------------------------------------------------
try:
    from geodatasets import get_path
    world = gpd.read_file(get_path("naturalearth.land"))
    print("World shapefile loaded via geodatasets.")
except Exception:
    try:
        world = gpd.read_file(
            "https://naciscdn.org/naturalearth/110m/physical/ne_110m_land.zip"
        )
        print("World shapefile loaded from naciscdn.")
    except Exception as e:
        raise RuntimeError(
            "Could not load world shapefile.\n"
            "Run:  pip install geodatasets\n"
            "Error: {e}"
        )

# ------------------------------------------------------------------
# 1. Load and prepare voyage data
# ------------------------------------------------------------------
df = pd.read_csv(INPUT_CSV)
df['lat_list'] = df['latitudes'].apply(ast.literal_eval)
df['lon_list'] = df['longitudes'].apply(ast.literal_eval)

# Derive starting coordinates from the list fields (origin_longitude/latitude are not in the labels file)
df['origin_latitude']  = df['lat_list'].apply(lambda x: x[0])
df['origin_longitude'] = df['lon_list'].apply(lambda x: x[0])

print(f"Loaded {len(df)} voyages, {df['cluster'].nunique()} clusters.")

# Print the number of voyages for each cluster, for reference when updating CLUSTER_LABELS
sizes = df['cluster'].value_counts().sort_index()
print("\nNumber of voyages per cluster (Reference for updating CLUSTER_LABELS):")
for c, n in sizes.items():
    print(f"  Cluster {c}: {n}")

# ------------------------------------------------------------------
# 2. Style
# ------------------------------------------------------------------
CLUSTER_COLORS = {
    0: '#E53935',
    1: '#1E88E5',
    2: '#FB8C00',
    3: '#43A047',
    4: '#8E24AA',
}

# ⚠️  TODO: After running cluster.m1.kmeans.py and inspecting the heatmap,
#           update the following label descriptions based on each cluster's behavioral characteristics.
CLUSTER_LABELS = {
    0: f'Cluster 0  (n={sizes.get(0, "?")})',
    1: f'Cluster 1  (n={sizes.get(1, "?")})',
    2: f'Cluster 2  (n={sizes.get(2, "?")})',
    3: f'Cluster 3  (n={sizes.get(3, "?")})',
    4: f'Cluster 4  (n={sizes.get(4, "?")})',
}
CLUSTER_TITLES = {
    0: 'Cluster 0',
    1: 'Cluster 1',
    2: 'Cluster 2',
    3: 'Cluster 3',
    4: 'Cluster 4',
}

# Light theme palette
THEME = dict(
    fig_bg   = 'white',
    ax_bg    = '#DDEEFF',
    land_fc  = '#D0DAE8',
    land_ec  = '#8899AA',
    grid_c   = '#888888',
    title_c  = '#111122',
    label_c  = '#333344',
    tick_c   = '#333344',
    spine_c  = '#8899AA',
    leg_bg   = 'white',
    leg_ec   = '#8899AA',
    leg_lc   = '#111122',
    leg_tc   = '#555566',
    nz_color = '#D32F2F',
    nz_tc    = '#111122',
    nz_stk   = 'white',
)

# ------------------------------------------------------------------
# 3. Coordinate helpers
# ------------------------------------------------------------------
def to_pacific_lon(lon):
    """Shift longitude to [20, 380] space so Pacific/NZ is central."""
    return lon + 360 if lon < 20 else lon

def unwrap_lons_pacific(lons):
    lons = [to_pacific_lon(l) for l in lons]
    for i in range(1, len(lons)):
        diff = lons[i] - lons[i-1]
        if diff > 180:
            lons[i] -= 360
        elif diff < -180:
            lons[i] += 360
    return lons

def shift_world(gdf):
    from shapely.affinity import translate
    from shapely.ops import unary_union
    import shapely

    parts = []
    for geom in gdf.geometry:
        if geom is None:
            continue
        east = geom.intersection(shapely.geometry.box(20, -90, 180, 90))
        west = geom.intersection(shapely.geometry.box(-180, -90, 20, 90))
        if not east.is_empty:
            parts.append(east)
        if not west.is_empty:
            parts.append(translate(west, xoff=360))
    combined = unary_union(parts)
    return gpd.GeoDataFrame(geometry=[combined], crs=gdf.crs)

print("Shifting world geometry to Pacific-centred projection...")
world_pac = shift_world(world)
print("Done.")

NZ_LON = to_pacific_lon(174.7)
NZ_LAT = -36.9

# ------------------------------------------------------------------
# 4. Core drawing helpers
# ------------------------------------------------------------------
def add_world(ax, xlim, ylim):
    t = THEME
    ax.set_facecolor(t['ax_bg'])
    world_pac.plot(ax=ax, color=t['land_fc'], edgecolor=t['land_ec'],
                    linewidth=0.4, zorder=2)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    x0, x1 = xlim
    y0, y1 = ylim
    for lon in range(int(x0)//30*30, int(x1)+30, 30):
        ax.axvline(lon, color=t['grid_c'], alpha=0.08, linewidth=0.5, zorder=1)
    for lat in range(int(y0)//30*30, int(y1)+30, 30):
        ax.axhline(lat, color=t['grid_c'], alpha=0.08, linewidth=0.5, zorder=1)
    ax.axhline(0, color=t['grid_c'], alpha=0.18, linewidth=0.8,
                linestyle='--', zorder=1)

def add_nz_marker(ax, fontsize=8):
    t = THEME
    ax.scatter(NZ_LON, NZ_LAT, color=t['nz_color'], s=160, zorder=8,
                marker='*', edgecolors='white', linewidths=0.8)
    ax.text(NZ_LON - 3, NZ_LAT + 3, 'NZ', color=t['nz_tc'],
            fontsize=fontsize, fontweight='bold', zorder=8, ha='right',
            path_effects=[pe.withStroke(linewidth=2, foreground=t['nz_stk'])])

def draw_voyages(ax, cluster_ids, alpha=0.20, lw=0.6):
    for cid in cluster_ids:
        sub   = df[df['cluster'] == cid]
        color = CLUSTER_COLORS[cid]
        for _, row in sub.iterrows():
            lons = unwrap_lons_pacific(row['lon_list'])
            lats = row['lat_list']
            if len(lons) < 2:
                continue
            ax.plot(lons, lats, color=color, alpha=alpha, linewidth=lw,
                    zorder=3, solid_capstyle='round')

def add_origin_dot(ax, cluster_id, s=90):
    """Plot a dot at the average position of the starting ports. Coordinates are derived from lat_list/lon_list[0]."""
    sub      = df[df['cluster'] == cluster_id]
    mean_lon = np.mean([to_pacific_lon(lon) for lon in sub['origin_longitude']])
    mean_lat = sub['origin_latitude'].mean()
    ax.scatter(mean_lon, mean_lat, color=CLUSTER_COLORS[cluster_id],
               s=s, zorder=6, edgecolors='white', linewidths=1.5)

def style_ax(ax, title='', fontsize=10):
    t = THEME
    ax.tick_params(colors=t['tick_c'], labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor(t['spine_c'])
    xticks = ax.get_xticks()
    xlabels = []
    for x in xticks:
        norm = x % 360
        if norm > 180:
            norm -= 360
        xlabels.append(f'{int(norm)}°')
    ax.set_xticklabels(xlabels, fontsize=7, color=t['tick_c'])
    yticks = ax.get_yticks()
    ax.set_yticklabels([f'{int(y)}°' for y in yticks],
                       fontsize=7, color=t['tick_c'])
    if title:
        ax.set_title(title, fontsize=fontsize, color=t['title_c'],
                     fontweight='bold', pad=6)

# ------------------------------------------------------------------
# 5. MAP 1 — NZ-centred world overview (all clusters)
# ------------------------------------------------------------------
print("\nDrawing world overview map...")

WORLD_XLIM = (20, 380)
WORLD_YLIM = (-75, 80)

fig, ax = plt.subplots(figsize=(20, 10), facecolor=THEME['fig_bg'])
add_world(ax, WORLD_XLIM, WORLD_YLIM)
draw_voyages(ax, range(5), alpha=0.18, lw=0.55)
for cid in range(5):
    add_origin_dot(ax, cid, s=90)
add_nz_marker(ax, fontsize=9)

handles = [mpatches.Patch(color=CLUSTER_COLORS[k], label=CLUSTER_LABELS[k])
           for k in range(5)]
legend = ax.legend(handles=handles, loc='lower left', fontsize=9,
                   framealpha=0.92, facecolor=THEME['leg_bg'],
                   edgecolor=THEME['leg_ec'], labelcolor=THEME['leg_lc'],
                   title='Voyage Clusters  (K-Means, k = 5)',
                   title_fontsize=9, borderpad=0.9, labelspacing=0.6)
legend.get_title().set_color(THEME['leg_tc'])

style_ax(ax, fontsize=13)
ax.set_xlabel('Longitude', color=THEME['label_c'], fontsize=10)
ax.set_ylabel('Latitude',  color=THEME['label_c'], fontsize=10)
fig.suptitle(
    'Voyage Cluster Map  —  K-Means  k = 5\n'
    'Historical voyages arriving at New Zealand ports  (2022–2024)',
    fontsize=13, fontweight='bold', color=THEME['title_c'], y=1.01
)
plt.tight_layout()
plt.savefig(OUTPUT_WORLD, dpi=200, bbox_inches='tight',
            facecolor=THEME['fig_bg'])
plt.close()
print(f"Saved: {OUTPUT_WORLD}")

# ------------------------------------------------------------------
# 6. Per-cluster bounding boxes
# ------------------------------------------------------------------
PAD = 8

def cluster_bbox(cluster_id, pad=PAD):
    sub  = df[df['cluster'] == cluster_id]
    lons = []
    lats = []
    for _, row in sub.iterrows():
        lons += [to_pacific_lon(l) for l in row['lon_list']]
        lats += list(row['lat_list'])
    lon_min = max(20,  min(lons) - pad)
    lon_max = min(380, max(lons) + pad)
    lat_min = max(-75, min(lats) - pad)
    lat_max = min(80,  max(lats) + pad)
    return (lon_min, lon_max), (lat_min, lat_max)

cluster_bboxes = {cid: cluster_bbox(cid) for cid in range(5)}

# ------------------------------------------------------------------
# 7. MAP 2 — Panel figure
# ------------------------------------------------------------------
print("Drawing panel map...")

fig = plt.figure(figsize=(22, 18), facecolor=THEME['fig_bg'])
gs  = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.15,
               height_ratios=[1.1, 1, 1])

ax_world = fig.add_subplot(gs[0, :])
add_world(ax_world, WORLD_XLIM, WORLD_YLIM)
draw_voyages(ax_world, range(5), alpha=0.18, lw=0.5)
for cid in range(5):
    add_origin_dot(ax_world, cid, s=80)
add_nz_marker(ax_world, fontsize=8)

handles = [mpatches.Patch(color=CLUSTER_COLORS[k], label=CLUSTER_LABELS[k])
           for k in range(5)]
legend = ax_world.legend(
    handles=handles, loc='lower left', fontsize=8.5, framealpha=0.92,
    facecolor=THEME['leg_bg'], edgecolor=THEME['leg_ec'],
    labelcolor=THEME['leg_lc'],
    title='Voyage Clusters  (K-Means, k = 5)',
    title_fontsize=8.5, borderpad=0.8, labelspacing=0.55
)
legend.get_title().set_color(THEME['leg_tc'])
style_ax(ax_world, title='(a)  All Clusters — Global Overview', fontsize=11)
ax_world.set_xlabel('Longitude', color=THEME['label_c'], fontsize=9)
ax_world.set_ylabel('Latitude',  color=THEME['label_c'], fontsize=9)

panel_positions = [(1, 0), (1, 1), (1, 2), (2, 0), (2, 1)]
panel_labels    = ['(b)', '(c)', '(d)', '(e)', '(f)']

for idx, cid in enumerate(range(5)):
    row, col = panel_positions[idx]
    ax = fig.add_subplot(gs[row, col])
    xlim, ylim = cluster_bboxes[cid]
    add_world(ax, xlim, ylim)
    draw_voyages(ax, [cid], alpha=0.30, lw=0.7)
    add_origin_dot(ax, cid, s=70)
    add_nz_marker(ax, fontsize=7)

    title_str = (f"{panel_labels[idx]}  {CLUSTER_TITLES[cid]}\n"
                 f"n = {(df['cluster']==cid).sum()}")
    style_ax(ax, title=title_str, fontsize=9)

    for spine in ax.spines.values():
        spine.set_edgecolor(CLUSTER_COLORS[cid])
        spine.set_linewidth(2)

fig.add_subplot(gs[2, 2]).set_visible(False)

fig.suptitle(
    'Voyage Cluster Maps  —  K-Means  k = 5\n'
    'Historical voyages arriving at New Zealand ports  (2022–2024)',
    fontsize=14, fontweight='bold', color=THEME['title_c'], y=1.01
)
plt.savefig(OUTPUT_PANEL, dpi=200, bbox_inches='tight',
            facecolor=THEME['fig_bg'])
plt.close()
print(f"Saved: {OUTPUT_PANEL}")

# ------------------------------------------------------------------
# 8. MAP 3 — Five standalone per-cluster maps
# ------------------------------------------------------------------
print("Drawing standalone cluster maps...")

for cid in range(5):
    xlim, ylim = cluster_bboxes[cid]
    fig, ax = plt.subplots(figsize=(12, 7), facecolor=THEME['fig_bg'])
    add_world(ax, xlim, ylim)
    draw_voyages(ax, [cid], alpha=0.30, lw=0.7)
    add_origin_dot(ax, cid, s=100)
    add_nz_marker(ax, fontsize=9)

    for spine in ax.spines.values():
        spine.set_edgecolor(CLUSTER_COLORS[cid])
        spine.set_linewidth(2.5)

    style_ax(ax, fontsize=12)
    ax.set_xlabel('Longitude', color=THEME['label_c'], fontsize=10)
    ax.set_ylabel('Latitude',  color=THEME['label_c'], fontsize=10)

    n = (df['cluster'] == cid).sum()
    fig.suptitle(
        f'{CLUSTER_TITLES[cid]}  —  n = {n}\n'
        'Historical voyages arriving at New Zealand ports  (2022–2024)',
        fontsize=12, fontweight='bold', color=THEME['title_c'], y=1.02
    )

    out = OUTPUT_CLUSTER_FMT.format(cid)
    plt.tight_layout()
    plt.savefig(out, dpi=200, bbox_inches='tight',
                facecolor=THEME['fig_bg'])
    plt.close()
    print(f"  Saved: {out}")

print("\nAll maps generated successfully.")
print("Output files:")
print(f"  {OUTPUT_WORLD}")
print(f"  {OUTPUT_PANEL}")
for cid in range(5):
    print(f"  {OUTPUT_CLUSTER_FMT.format(cid)}")