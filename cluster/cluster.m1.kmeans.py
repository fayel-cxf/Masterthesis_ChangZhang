# =============================================================================
# cluster.m1.kmeans.py
# K-Means 聚类脚本 —— 基于 m1 特征工程方案的预处理输出
#
# INPUT:  data/clustering/cluster.m1.prep.csv       (由 cluster.m1.py 生成)
# OUTPUT:
#   - data/clustering/cluster.m1.labels.k5.csv      (原始数据 + cluster 列)
#   - graphics/cluster.pics/m1.kmeans.elbow.png
#   - graphics/cluster.pics/m1.kmeans.silhouette.png
#   - graphics/cluster.pics/m1.kmeans.heatmap_k3.png
#   - graphics/cluster.pics/m1.kmeans.heatmap_k5.png
#
# 本脚本是聚类 pipeline 的第二步，不包含任何特征工程逻辑。
# 所有预处理已由 cluster.m1.py 完成。
# =============================================================================
#
# -----------------------------------------------------------------------------
# 聚类方案说明
# -----------------------------------------------------------------------------
#
# [k 的选择]
#   Silhouette 最优值出现在 k=3（约 0.30），但 k=3 时超过 60% 的航次
#   集中于同一个簇，失去了对 Markov Chain 条件生成有价值的行为区分。
#   最终选择 k=5：五个簇对应五类可解释的航次行为模式，为下游
#   Markov Chain 的条件生成提供更细粒度的结构信息。
#   k=3 和 k=5 的 heatmap 均输出，供论文对比展示。
#
# [初始化]
#   n_init=20：每次聚类运行 20 组不同初始化，取 inertia 最小的结果，
#   降低 k-means 对初始中心点敏感性的影响。
#
# [输出标签文件]
#   读取原始数据 observed_voyages_cleaned.csv，将 cluster 标签拼接回去，
#   保留所有原始字段，确保后续 Markov Chain 脚本可直接使用。
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ------------------------------------------------------------------
# 路径配置
# ------------------------------------------------------------------
BASE_DIR           = Path(__file__).resolve().parent.parent.parent
INPUT_PREP         = BASE_DIR / "data/clustering/cluster.m1.prep.csv"
INPUT_RAW          = BASE_DIR / "data/clustering/observed_voyages_cleaned.csv"
OUTPUT_LABELS      = BASE_DIR / "data/clustering/cluster.m1.labels.k5.csv"
OUTPUT_ELBOW       = BASE_DIR / "graphics/cluster.pics/m1.kmeans.elbow.png"
OUTPUT_SILHOUETTE  = BASE_DIR / "graphics/cluster.pics/m1.kmeans.silhouette.png"
OUTPUT_HEATMAP_K3  = BASE_DIR / "graphics/cluster.pics/m1.kmeans.heatmap_k3.png"
OUTPUT_HEATMAP_K5  = BASE_DIR / "graphics/cluster.pics/m1.kmeans.heatmap_k5.png"

K_MIN        = 2
K_MAX        = 15
RANDOM_STATE = 42
HEATMAP_KS   = [3, 5]
CHOSEN_K     = 5

for path in [OUTPUT_ELBOW.parent, OUTPUT_LABELS.parent]:
    path.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------
# 读取预处理特征矩阵
# ------------------------------------------------------------------
df_prep = pd.read_csv(INPUT_PREP)
X = df_prep.values
feature_names = list(df_prep.columns)
print(f"[读取] 特征矩阵: {X.shape[0]} 行 × {X.shape[1]} 列")

# ------------------------------------------------------------------
# 计算各 k 的 inertia 和 silhouette，缓存 HEATMAP_KS 的模型
# ------------------------------------------------------------------
k_values    = list(range(K_MIN, K_MAX + 1))
inertias    = []
silhouettes = []
km_cache    = {}

print("\n[K-Means] 开始扫描 k 值...")
for k in k_values:
    km     = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=20)
    labels = km.fit_predict(X)
    inertias.append(km.inertia_)
    sil = silhouette_score(X, labels, sample_size=min(2000, X.shape[0]),
                           random_state=RANDOM_STATE)
    silhouettes.append(sil)
    print(f"  k={k:2d}  inertia={km.inertia_:,.1f}  silhouette={sil:.4f}")
    if k in HEATMAP_KS:
        km_cache[k] = km

# ------------------------------------------------------------------
# Elbow plot
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(k_values, inertias, marker="o", linewidth=2, color="steelblue")
ax.set_xlabel("Number of clusters  k", fontsize=12)
ax.set_ylabel("Inertia (within-cluster SSE)", fontsize=12)
ax.set_title("Elbow Method — K-Means (m1 features)", fontsize=13)
ax.set_xticks(k_values)
ax.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(OUTPUT_ELBOW, dpi=150)
plt.close()
print(f"\n[图表] Elbow plot 已保存至 {OUTPUT_ELBOW}")

# ------------------------------------------------------------------
# Silhouette plot
# ------------------------------------------------------------------
best_k   = k_values[int(np.argmax(silhouettes))]
best_sil = max(silhouettes)

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(k_values, silhouettes, marker="o", linewidth=2, color="darkorange")
ax.axvline(best_k, linestyle="--", color="gray", alpha=0.7)
ax.axvline(CHOSEN_K, linestyle="--", color="steelblue", alpha=0.7)
ax.annotate(f"silhouette best\nk={best_k} ({best_sil:.4f})",
            xy=(best_k, best_sil),
            xytext=(best_k + 0.4, best_sil - 0.005),
            fontsize=9, color="gray")
ax.annotate(f"chosen k={CHOSEN_K}",
            xy=(CHOSEN_K, silhouettes[CHOSEN_K - K_MIN]),
            xytext=(CHOSEN_K + 0.4, silhouettes[CHOSEN_K - K_MIN] - 0.005),
            fontsize=9, color="steelblue")
ax.set_xlabel("Number of clusters  k", fontsize=12)
ax.set_ylabel("Silhouette Score", fontsize=12)
ax.set_title("Silhouette Score — K-Means (m1 features)", fontsize=13)
ax.set_xticks(k_values)
ax.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(OUTPUT_SILHOUETTE, dpi=150)
plt.close()
print(f"[图表] Silhouette plot 已保存至 {OUTPUT_SILHOUETTE}")

# ------------------------------------------------------------------
# 汇总表
# ------------------------------------------------------------------
print("\n--- K-Means 扫描结果 ---")
print(f"{'k':>4}  {'Inertia':>14}  {'Silhouette':>12}")
print("-" * 35)
for k, ine, sil in zip(k_values, inertias, silhouettes):
    tag = ""
    if k == best_k:
        tag += "  ← silhouette best"
    if k == CHOSEN_K and CHOSEN_K != best_k:
        tag += "  ← chosen"
    print(f"{k:>4}  {ine:>14,.1f}  {sil:>12.4f}{tag}")

# ------------------------------------------------------------------
# Heatmap（k=3 和 k=5）
# ------------------------------------------------------------------
FEATURE_LABELS = [
    "n_ports", "total_span_days", "avg_dwell_time", "dwell_ratio",
    "DWT", "total_distance", "max_dist/leg",
    "origin_lat", "lon_sin", "lon_cos",
    "lon_range", "hemisphere_cross", "n_territories",
    "type_Bulker", "type_Container", "type_GenCargo", "type_Other",
    "type_Passenger", "type_Reefer", "type_RoRo", "type_Tanker",
]

def plot_heatmap(km, k, output_path):
    centroids  = km.cluster_centers_
    labels_fit = km.labels_
    sizes      = pd.Series(labels_fit).value_counts().sort_index()
    n_total    = len(labels_fit)

    row_labels = [
        f"Cluster {c}  (n={sizes[c]}, {100*sizes[c]/n_total:.1f}%)"
        for c in range(k)
    ]

    fig, ax = plt.subplots(figsize=(15, max(4, 1.4 * k + 1.5)))
    im = ax.imshow(centroids, aspect="auto", cmap="RdBu_r", vmin=-2.5, vmax=2.5)
    ax.set_xticks(range(len(FEATURE_LABELS)))
    ax.set_xticklabels(FEATURE_LABELS, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(k))
    ax.set_yticklabels(row_labels, fontsize=10)
    ax.set_title(
        f"Cluster Centroid Heatmap  (k={k})  —  standardised values\n"
        "red = high, blue = low, white = near mean",
        fontsize=11, pad=10
    )
    for r in range(k):
        for c in range(len(FEATURE_LABELS)):
            val = centroids[r, c]
            txt_color = "white" if abs(val) > 1.5 else "black"
            ax.text(c, r, f"{val:.2f}", ha="center", va="center",
                    fontsize=7.5, color=txt_color)
    plt.colorbar(im, ax=ax, label="Standardised centroid value", shrink=0.6)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[图表] Heatmap (k={k}) 已保存至 {output_path}")

plot_heatmap(km_cache[3], 3, OUTPUT_HEATMAP_K3)
plot_heatmap(km_cache[5], 5, OUTPUT_HEATMAP_K5)

# ------------------------------------------------------------------
# 拟合最终模型（k=CHOSEN_K），将标签拼回原始数据
# ------------------------------------------------------------------
km_final     = km_cache[CHOSEN_K]
final_labels = km_final.labels_

df_raw = pd.read_csv(INPUT_RAW)

# 对齐行数（cluster.m1.py 可能删除了含缺失值的行）
if len(df_raw) != len(final_labels):
    print(f"[警告] 原始数据 {len(df_raw)} 行，特征矩阵 {len(final_labels)} 行，"
          f"以特征矩阵行数为准截断原始数据")
    df_raw = df_raw.iloc[:len(final_labels)].reset_index(drop=True)

df_raw["cluster"] = final_labels
df_raw.to_csv(OUTPUT_LABELS, index=False)

print(f"\n[完成] 最终模型 k={CHOSEN_K}")
print("各聚类航次数：")
sizes = pd.Series(final_labels).value_counts().sort_index()
for c, n in sizes.items():
    print(f"  Cluster {c}: {n} 条 ({100*n/len(final_labels):.1f}%)")
print(f"\n[输出] 带标签数据已保存至 {OUTPUT_LABELS}")
