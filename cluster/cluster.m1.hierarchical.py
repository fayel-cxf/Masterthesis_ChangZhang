# =============================================================================
# cluster.m1.hierarchical.py
# 层次聚类脚本 —— 基于 m1 特征工程方案的预处理输出
#
# INPUT:  data/clustering/cluster.m1.prep.csv          (由 cluster.m1.py 生成)
# OUTPUT:
#   - data/clustering/cluster.m1.labels.hierarchical.csv  (原始数据 + cluster 列)
#   - graphics/cluster.pics/m1.hierarchical.dendrogram.png
#   - graphics/cluster.pics/m1.hierarchical.silhouette.png
#   - graphics/cluster.pics/m1.hierarchical.heatmap_k3.png
#   - graphics/cluster.pics/m1.hierarchical.heatmap_k5.png
#
# 本脚本是聚类 pipeline 的备选方法，与 cluster.m1.kmeans.py 使用相同输入，
# 供方法对比和论文讨论使用。不包含任何特征工程逻辑。
# =============================================================================
#
# -----------------------------------------------------------------------------
# 方法说明
# -----------------------------------------------------------------------------
#
# [算法选择]
#   使用 Ward 连接（Ward linkage）的凝聚层次聚类（Agglomerative Clustering）。
#   Ward 连接在每步合并时最小化簇内方差的增量，与 k-means 的目标函数一致
#   （最小化 within-cluster SSE），因此两者结果具有可比性。
#   适用于本数据集的原因：
#     - 航次数据为连续数值特征，欧氏距离有意义
#     - Ward 连接对噪声和异常值的鲁棒性优于 single/complete linkage
#     - 可通过 dendrogram 直观展示簇间合并距离，辅助 k 的选择
#
# [k 的选择]
#   与 k-means 保持一致，输出 k=3 和 k=5 两种方案的 heatmap 供对比。
#   CHOSEN_K 默认设为 5，与 k-means 最终选择对齐，便于两种方法的结果比较。
#
# [Dendrogram 截断显示]
#   完整 dendrogram 在 2000+ 样本下难以阅读，使用 truncate_mode='lastp'
#   仅展示最后 DENDROGRAM_LASTP 次合并，保留整体树形结构的可读性。
#   水平虚线标注 k=3 和 k=5 对应的截断高度，直观显示分割位置。
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist

# ------------------------------------------------------------------
# 路径配置
# ------------------------------------------------------------------
BASE_DIR          = Path(__file__).resolve().parent.parent.parent
INPUT_PREP        = BASE_DIR / "data/clustering/cluster.m1.prep.csv"
INPUT_RAW         = BASE_DIR / "data/clustering/observed_voyages_cleaned.csv"
OUTPUT_LABELS     = BASE_DIR / "data/clustering/cluster.m1.labels.hierarchical.csv"
OUTPUT_DENDROGRAM = BASE_DIR / "graphics/cluster.pics/m1.hierarchical.dendrogram.png"
OUTPUT_SILHOUETTE = BASE_DIR / "graphics/cluster.pics/m1.hierarchical.silhouette.png"
OUTPUT_HEATMAP_K3 = BASE_DIR / "graphics/cluster.pics/m1.hierarchical.heatmap_k3.png"
OUTPUT_HEATMAP_K5 = BASE_DIR / "graphics/cluster.pics/m1.hierarchical.heatmap_k5.png"

K_MIN             = 2
K_MAX             = 15
CHOSEN_K          = 5
HEATMAP_KS        = [3, 5]
DENDROGRAM_LASTP  = 50    # dendrogram 展示最后 50 次合并

for path in [OUTPUT_DENDROGRAM.parent, OUTPUT_LABELS.parent]:
    path.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------
# 读取预处理特征矩阵
# ------------------------------------------------------------------
df_prep = pd.read_csv(INPUT_PREP)
X = df_prep.values
feature_names = list(df_prep.columns)
print(f"[读取] 特征矩阵: {X.shape[0]} 行 × {X.shape[1]} 列")

# ------------------------------------------------------------------
# 计算完整 linkage 矩阵（Ward 方法，供 dendrogram 和标签提取使用）
# ------------------------------------------------------------------
print("\n[层次聚类] 计算 Ward linkage 矩阵...")
Z = linkage(X, method="ward", metric="euclidean")
print("[层次聚类] Linkage 矩阵计算完成")

# ------------------------------------------------------------------
# Dendrogram
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(14, 6))
dendrogram(
    Z,
    truncate_mode="lastp",
    p=DENDROGRAM_LASTP,
    leaf_rotation=90,
    leaf_font_size=8,
    ax=ax,
    color_threshold=0,
    above_threshold_color="steelblue",
)

# 标注 k=3 和 k=5 的截断高度
for k, color, label in [(3, "darkorange", "k=3"), (5, "crimson", "k=5")]:
    # 从 linkage 矩阵中找到对应截断高度
    cut_height = Z[-(k - 1), 2]
    ax.axhline(cut_height, linestyle="--", color=color, alpha=0.8,
               label=f"{label}  (height={cut_height:.1f})")

ax.set_xlabel("Sample index (truncated to last 50 merges)", fontsize=11)
ax.set_ylabel("Ward linkage distance", fontsize=11)
ax.set_title(f"Hierarchical Clustering Dendrogram — Ward Linkage (m1 features)\n"
             f"(showing last {DENDROGRAM_LASTP} merges)", fontsize=12)
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig(OUTPUT_DENDROGRAM, dpi=150, bbox_inches="tight")
plt.close()
print(f"[图表] Dendrogram 已保存至 {OUTPUT_DENDROGRAM}")

# ------------------------------------------------------------------
# 扫描 k 值 —— silhouette score
# ------------------------------------------------------------------
k_values    = list(range(K_MIN, K_MAX + 1))
silhouettes = []
label_cache = {}

print("\n[层次聚类] 扫描 k 值...")
for k in k_values:
    labels = fcluster(Z, k, criterion="maxclust") - 1   # 转为 0-indexed
    sil    = silhouette_score(X, labels, sample_size=min(2000, X.shape[0]),
                              random_state=42)
    silhouettes.append(sil)
    print(f"  k={k:2d}  silhouette={sil:.4f}")
    if k in HEATMAP_KS or k == CHOSEN_K:
        label_cache[k] = labels

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
ax.set_title("Silhouette Score — Hierarchical Clustering / Ward (m1 features)", fontsize=13)
ax.set_xticks(k_values)
ax.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(OUTPUT_SILHOUETTE, dpi=150)
plt.close()
print(f"\n[图表] Silhouette plot 已保存至 {OUTPUT_SILHOUETTE}")

# ------------------------------------------------------------------
# 汇总表
# ------------------------------------------------------------------
print("\n--- 层次聚类扫描结果 ---")
print(f"{'k':>4}  {'Silhouette':>12}")
print("-" * 20)
for k, sil in zip(k_values, silhouettes):
    tag = ""
    if k == best_k:
        tag += "  ← silhouette best"
    if k == CHOSEN_K and CHOSEN_K != best_k:
        tag += "  ← chosen"
    print(f"{k:>4}  {sil:>12.4f}{tag}")

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

def plot_heatmap(labels, k, output_path):
    # 计算各簇中心（各簇样本均值）
    centroids = np.array([X[labels == c].mean(axis=0) for c in range(k)])
    sizes     = pd.Series(labels).value_counts().sort_index()
    n_total   = len(labels)

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
        f"Cluster Centroid Heatmap  (k={k})  —  Ward Hierarchical\n"
        "red = high, blue = low, white = near mean",
        fontsize=11, pad=10
    )
    for r in range(k):
        for c_idx in range(len(FEATURE_LABELS)):
            val = centroids[r, c_idx]
            txt_color = "white" if abs(val) > 1.5 else "black"
            ax.text(c_idx, r, f"{val:.2f}", ha="center", va="center",
                    fontsize=7.5, color=txt_color)
    plt.colorbar(im, ax=ax, label="Standardised centroid value", shrink=0.6)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[图表] Heatmap (k={k}) 已保存至 {output_path}")

plot_heatmap(label_cache[3], 3, OUTPUT_HEATMAP_K3)
plot_heatmap(label_cache[5], 5, OUTPUT_HEATMAP_K5)

# ------------------------------------------------------------------
# 将最终标签（k=CHOSEN_K）拼回原始数据
# ------------------------------------------------------------------
final_labels = label_cache[CHOSEN_K]

df_raw = pd.read_csv(INPUT_RAW)
if len(df_raw) != len(final_labels):
    print(f"[警告] 原始数据 {len(df_raw)} 行，特征矩阵 {len(final_labels)} 行，"
          f"以特征矩阵行数为准截断原始数据")
    df_raw = df_raw.iloc[:len(final_labels)].reset_index(drop=True)

df_raw["cluster"] = final_labels
df_raw.to_csv(OUTPUT_LABELS, index=False)

print(f"\n[完成] 最终方案 k={CHOSEN_K}（Ward 层次聚类）")
print("各聚类航次数：")
sizes = pd.Series(final_labels).value_counts().sort_index()
for c, n in sizes.items():
    print(f"  Cluster {c}: {n} 条 ({100*n/len(final_labels):.1f}%)")
print(f"\n[输出] 带标签数据已保存至 {OUTPUT_LABELS}")
