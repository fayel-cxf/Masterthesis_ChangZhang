# =============================================================================
# cluster.m1.py
# 特征工程脚本 —— 从原始航次数据生成聚类特征矩阵
#
# INPUT:  data/clustering/observed_voyages_cleaned.csv   (2432 行 × 21 列)
# OUTPUT: data/clustering/cluster.m1.prep.csv            (预处理后特征矩阵)
#
# 本脚本是聚类 pipeline 的第一步（m1 = 特征工程方案一）。
# 输出文件 cluster.m1.prep.csv 供 cluster.m1.kmeans.py 直接读入，进行 k-means 聚类。
#
# 本脚本不依赖任何外部预处理文件，所有派生特征均从原始字段计算。
# =============================================================================
#
# -----------------------------------------------------------------------------
# 所有处理步骤总览
# -----------------------------------------------------------------------------
#
# [Step 1]  解析列表型字段
#           latitudes / longitudes / dwells_hours / transits_hours
#           在 CSV 中以字符串形式存储（如 "[-6.09, -19.09, ...]"），
#           使用 ast.literal_eval 解析为 Python list of float。
#
# [Step 2]  计算派生特征
#
#   [2a] 时间特征
#        - avg_dwell_time       : 每次停港的平均停留时间（小时）= mean(dwells_hours)
#        - voyage_duration_total: 总在途时间（小时）= sum(transits_hours)
#
#   [2b] 地理距离特征（Haversine 大圆距离，单位 km）
#        - max_distance_per_leg : 单段最长距离
#        - total_distance       : 所有腿距离之和
#        注：avg_distance_per_leg 与 total_distance 相关系数 r = 0.877，
#        属冗余变量，不计算也不纳入特征。
#
#   [2c] 起始地理位置
#        - origin_latitude      : 起始港纬度（latitudes[0]）
#        - origin_longitude     : 起始港经度（longitudes[0]），后续循环编码
#
#   [2d] 航线地理跨度
#        - longitude_range      : 经度跨度 = max(longitudes) − min(longitudes)
#        - hemisphere_crossing  : 是否跨赤道（0/1），
#                                 min(latitudes) < 0 且 max(latitudes) > 0 时为 1
#        注：destination_latitude（所有航次终点均为新西兰，方差近零）
#        和 geographic_range（与 origin_latitude r = 0.77，线性冗余）均不纳入。
#
#   [2e] 停港行为
#        - dwell_ratio          : 停港时间占总航程时间的比例
#                                 = (avg_dwell_time × n_ports) /
#                                   (avg_dwell_time × n_ports + voyage_duration_total)
#
#   [2f] 途经国家/地区数
#        - n_territories        : port_territories 去重后的数量
#
# [Step 3]  One-hot 编码船舶类型
#           nbic_type_group → type_Bulker, type_Container, type_General Cargo,
#                              type_Other, type_Passenger, type_Reefer,
#                              type_RoRo, type_Tanker（共 8 列，值为 0/1）
#
# [Step 4]  删除冗余时间变量
#           - voyage_duration_total : 与 total_span_days 相关系数 r = 0.821，冗余
#           - avg_transit_time      : 与 avg_distance_per_leg r = 0.790，
#                                     且本身不在特征集中
#           仅保留 total_span_days（已在原始数据中），avg_dwell_time，dwell_ratio。
#
# [Step 5]  经度循环编码
#           经度是 [−180, 180] 上的循环变量，直接用原始值会导致欧氏距离失真
#           （如 170°E 与 170°W 原始差值 340，实际地理距离仅约 20°）。
#           将 origin_longitude 替换为：
#             lon_sin = sin(longitude × π / 180)
#             lon_cos = cos(longitude × π / 180)
#           删除原始 origin_longitude 列。
#
# [Step 6]  对数变换（在标准化之前）
#           仅对以下两列施加 log1p 变换，理由如下：
#           - avg_dwell_time : 偏度 = 8.07，存在极端值（最大 1224 小时，
#                              95 分位数仅 135 小时）。标准化无法消除极端值
#                              对欧氏距离的主导效应，log1p 使分布趋于对称。
#           - DWT            : 船型间吨位跨越量级（散货船 >100,000 t，
#                              小型杂货船 ~2,000 t）。log 变换确保大/中型船
#                              与中/小型船之间的差异在距离度量中权重均等。
#           距离类变量（total_distance, max_distance_per_leg, longitude_range）
#           不做 log 变换——其绝对差异正是聚类应捕捉的航程规模信号；
#           StandardScaler 已足以消除量纲差异。
#
# [Step 7]  标准化（Z-score）
#           对全部 13 个连续/计数特征使用 StandardScaler（均值 0，方差 1），
#           消除量纲差异，避免单一变量主导欧氏距离。
#           8 个 one-hot 船型列保持 0/1 不变，不参与标准化，
#           最后拼接到标准化后的连续特征矩阵末尾。
#
# [Step 8]  删除含缺失值的行
#           若派生特征计算失败（如空列表），删除对应行并打印警告。
#
# -----------------------------------------------------------------------------
# 最终特征集（13 个连续/计数特征 + 8 个 one-hot = 21 列）
# -----------------------------------------------------------------------------
#   连续/计数特征（经标准化）：
#     n_ports               总停港次数
#     total_span_days       总航程天数
#     avg_dwell_time *      每次停港平均时长（小时）        [log1p]
#     dwell_ratio           停港时间占总航程比例
#     DWT *                 压舱水回归模型核心变量（吨）    [log1p]
#     total_distance        总航行距离（km）
#     max_distance_per_leg  单段最长距离（km）
#     origin_latitude       起始港纬度
#     lon_sin               起始港经度 sin 分量
#     lon_cos               起始港经度 cos 分量
#     longitude_range       经度跨度
#     hemisphere_crossing   是否跨赤道（0/1，标准化后连续）
#     n_territories         途经国家/地区数
#
#   One-hot 船型（不标准化）：
#     type_Bulker, type_Container, type_General Cargo, type_Other,
#     type_Passenger, type_Reefer, type_RoRo, type_Tanker
# =============================================================================

import ast
import numpy as np
import pandas as pd
from math import radians, cos, sin, asin, sqrt
from pathlib import Path
from sklearn.preprocessing import StandardScaler

# ------------------------------------------------------------------
# 路径配置
# ------------------------------------------------------------------
BASE_DIR   = Path(__file__).resolve().parent.parent.parent
INPUT_PATH = BASE_DIR / "data/clustering/observed_voyages_cleaned.csv"
OUTPUT_PATH = BASE_DIR / "data/clustering/cluster.m1.prep.csv"

# ------------------------------------------------------------------
# 读取原始数据
# ------------------------------------------------------------------
df = pd.read_csv(INPUT_PATH)
print(f"[读取] {df.shape[0]} 行 × {df.shape[1]} 列")

# ------------------------------------------------------------------
# Step 1: 解析列表型字段
# ------------------------------------------------------------------
def parse_list_col(series):
    return series.apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

df["latitudes"]      = parse_list_col(df["latitudes"])
df["longitudes"]     = parse_list_col(df["longitudes"])
df["dwells_hours"]   = parse_list_col(df["dwells_hours"])
df["transits_hours"] = parse_list_col(df["transits_hours"])
print("[Step 1] 列表型字段解析完成")

# ------------------------------------------------------------------
# Step 2: 计算派生特征
# ------------------------------------------------------------------

# [2a] 时间特征
df["avg_dwell_time"]         = df["dwells_hours"].apply(
    lambda x: np.mean(x) if len(x) > 0 else np.nan)
df["voyage_duration_total"]  = df["transits_hours"].apply(np.sum)

# [2b] 距离特征（Haversine）
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

# [2c] 起始地理位置
df["origin_latitude"]  = df["latitudes"].apply(lambda x: x[0])
df["origin_longitude"] = df["longitudes"].apply(lambda x: x[0])

# [2d] 航线地理跨度
df["longitude_range"]     = df["longitudes"].apply(lambda x: max(x) - min(x))
df["hemisphere_crossing"] = df["latitudes"].apply(
    lambda x: 1.0 if (min(x) < 0 and max(x) > 0) else 0.0)

# [2e] 停港比例
total_time = df["avg_dwell_time"] * df["n_ports"] + df["voyage_duration_total"]
df["dwell_ratio"] = (df["avg_dwell_time"] * df["n_ports"]) / total_time.replace(0, np.nan)

# [2f] 途经国家/地区数
df["n_territories"] = df["port_territories"].apply(
    lambda x: len(set(ast.literal_eval(x))) if isinstance(x, str) else np.nan)

print("[Step 2] 派生特征计算完成")

# ------------------------------------------------------------------
# Step 3: One-hot 编码船舶类型
# ------------------------------------------------------------------
VESSEL_TYPES = ["Bulker", "Container", "General Cargo", "Other",
                "Passenger", "Reefer", "RoRo", "Tanker"]
for vt in VESSEL_TYPES:
    df[f"type_{vt}"] = (df["nbic_type_group"] == vt).astype(int)
print("[Step 3] 船型 one-hot 编码完成")

# ------------------------------------------------------------------
# Step 4: 删除冗余时间变量
#         voyage_duration_total（r=0.821 与 total_span_days）
#         仅用于计算 dwell_ratio，计算完成后不再保留
# ------------------------------------------------------------------
df.drop(columns=["voyage_duration_total"], inplace=True)
print("[Step 4] 冗余时间变量已删除")

# ------------------------------------------------------------------
# Step 5: 经度循环编码
# ------------------------------------------------------------------
df["lon_sin"] = np.sin(np.radians(df["origin_longitude"]))
df["lon_cos"] = np.cos(np.radians(df["origin_longitude"]))
df.drop(columns=["origin_longitude"], inplace=True)
print("[Step 5] 经度循环编码完成")

# ------------------------------------------------------------------
# Step 6: Log1p 变换
# ------------------------------------------------------------------
for col in ["avg_dwell_time", "DWT"]:
    df[col] = np.log1p(df[col])
print("[Step 6] log1p 变换完成（avg_dwell_time, DWT）")

# ------------------------------------------------------------------
# Step 7: 组合特征矩阵并标准化
# ------------------------------------------------------------------
CONTINUOUS_COLS = [
    "n_ports", "total_span_days", "avg_dwell_time", "dwell_ratio",
    "DWT", "total_distance", "max_distance_per_leg",
    "origin_latitude", "lon_sin", "lon_cos",
    "longitude_range", "hemisphere_crossing", "n_territories",
]
ONEHOT_COLS = [f"type_{vt}" for vt in VESSEL_TYPES]

df_feat = df[CONTINUOUS_COLS + ONEHOT_COLS].copy()

# Step 8: 删除含缺失值的行
mask_valid = df_feat.notna().all(axis=1)
n_dropped = (~mask_valid).sum()
if n_dropped > 0:
    print(f"[Step 8] 警告：删除 {n_dropped} 行含缺失值的记录")
df_feat = df_feat[mask_valid].reset_index(drop=True)

# 标准化连续特征
scaler = StandardScaler()
df_feat[CONTINUOUS_COLS] = scaler.fit_transform(df_feat[CONTINUOUS_COLS])
print("[Step 7] 标准化完成")

# ------------------------------------------------------------------
# 保存
# ------------------------------------------------------------------
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
df_feat.to_csv(OUTPUT_PATH, index=False)
print(f"\n[完成] 特征矩阵 {df_feat.shape[0]} 行 × {df_feat.shape[1]} 列")
print(f"[输出] {OUTPUT_PATH}")
print(f"列名: {list(df_feat.columns)}")
