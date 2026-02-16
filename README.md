# Project Urban Noise  
## Block-Level Urban Form Feature Extraction & Heatmap Analysis (NYC)

---

## 项目简介

本项目构建了一个基于纽约市 MapPLUTO 数据的区块级（block-level）城市形态特征提取与可视化管线。

目标是：

- 将地块（parcel）数据聚合为区块尺度特征  
- 计算关键城市形态指标（建筑密度、容积率、高度异质性等）  
- 生成可直接用于报告与展示的热力图  
- 为后续噪声模型或空间分析提供标准化输入数据  

---

## 数据来源

### 1. MapPLUTO 2025v4（GDB 格式）

主要字段：

- Borough  
- Block  
- BldgArea  
- NumFloors  
- BuiltFAR  
- BBL  

### 2. 区块几何数据

- `data/processed/blocks.geojson`  
- 投影坐标系：EPSG:2263  

---

## 区块级指标定义

### 1. 容积率（FAR）

```
FAR = 区块总建筑面积 / 区块面积
```

---

### 2. 建筑密度（Footprint Proxy）

由于缺少精确 footprint 数据，采用代理方法：

```
建筑密度 = (总建筑面积 / 平均层数) / 区块面积
```

---

### 3. 建筑高度

假设：

```
1 层 = 10 ft
```

计算方式：

- height_mean = 平均层数 × 10  
- height_std = 层数标准差 × 10  

---

### 4. 天空开阔度（Proxy）

```
sky_openness_proxy = 1 − building_density
```

范围限制在 [0, 1]

---

## 可视化方法

生成两张区块级热力图：

- 建筑密度热力图  
- 高度异质性热力图  

可视化策略：

- 红色代表数值高（YlOrRd 色带）  
- 分位数分级（k = 11）  
- 99% 分位截断，避免极端值影响色阶  
- 去除边框线条，适合报告展示  

输出文件：

```
figures/heatmap_building_density_quantiles_red.png
figures/heatmap_height_heterogeneity_quantiles_red.png
```

---

## 输出结果

核心输出文件：

```
data/processed/block_features.csv
data/processed/block_features.geojson
```

区块总数：28,790  
MapPLUTO 匹配率：96.97%

---

## 运行方式

### 1. 计算区块特征

```
python -m src.compute_block_features --config config.yaml
```

### 2. 生成热力图

```
python -m src.plot_heatmaps --config config.yaml
```
