# Project_Urban_Noise (SYSEN 5900)


### 0) 本次文档更新范围
- 本次协作按要求仅更新 `README.md`，不改动 `src/`、`config.example.yaml`、`requirements.txt` 与任何数据文件。
- 目标是让组员打开 README 即可按步骤跑出 blocks、block_features 与两张红色热力图。

### 1) 项目简介
本项目面向 **Project_Urban_Noise（SYSEN 5900）** 的组内数据生产环节：将 NYC 城市地理数据（以 MapPLUTO 为核心）转换为 **block-level urban form features（街区级形态特征）**，用于后续整合建模与分析，并交付给 **熊梓杰 + 舒英一** 进行整合。当前管线完成了：MapPLUTO parcel/lot 读取 → block 几何构建（dissolve）→ block 指标计算（CSV/GeoJSON）→ 红色高值分位数热力图输出。

### 2) 数据输入
> 原始数据只放本地 `data/raw/`，不要提交到 Git。

核心输入（当前版本主要依赖）：
- `data/raw/MapPLUTO25v4_unclipped.gdb`（layer: `MapPLUTO_25v4_unclipped`）

可选输入（当前核心指标不依赖）：
- `data/raw/BUILDING_20260209.csv`
- `data/raw/bldg_3d_metrics.csv`
- `data/raw/Centerline_20260209.csv`

运行配置文件：
- `config.yaml`（从 `config.example.yaml` 复制并按需修改）

### 3) 输出结果
执行完成后可得到：
- `data/processed/blocks.geojson`
- `data/processed/block_features.geojson`
- `data/processed/block_features.csv`
- `figures/heatmap_building_density_quantiles_red.png`
- `figures/heatmap_height_heterogeneity_quantiles_red.png`

### 4) 指标定义（block 级）
设 block 面积为 `block_area_ft2`，并由 parcel 聚合得到 `bldgarea_sum`（总建筑面积 proxy）与 `numfloors_mean`。

- `building_density`（覆盖率/密度 proxy）  
  - `footprint_area_sum = bldgarea_sum / max(numfloors_mean, 1)`  
  - `building_density = footprint_area_sum / block_area_ft2`
- `FAR`  
  - `FAR = gross_floor_area_sum / block_area_ft2`，其中 `gross_floor_area_sum = bldgarea_sum`
- `height_mean`、`height_std`（高度异质性）  
  - `height_mean = numfloors_mean * 10 ft`  
  - `height_std = numfloors_std * 10 ft`
- `sky_openness_proxy`  
  - `sky_openness_proxy = clip(1 - building_density, 0, 1)`
- `street_hw_ratio`  
  - 当前稳定版 **未纳入主输出**（可选/后续扩展；若需要将基于中心线宽度或代理方式单独计算并并入）

### 5) 如何运行（按顺序）
> 下面命令可直接在仓库根目录运行（Windows PowerShell 友好）。

1. 准备配置：
```powershell
Copy-Item config.example.yaml config.yaml
```

2. 生成 block 几何：
```powershell
python -m src.build_blocks --config config.yaml
```

3. 计算 block 特征：
```powershell
python -m src.compute_block_features --config config.yaml
```

4. 生成热力图（红色=高值，分位数分级，99% 截断）：
```powershell
python -m src.plot_heatmaps --config config.yaml
```

### 6) 性能与耗时提示
- **MapPLUTO GDB 读取与 dissolve 可能较慢**（机器配置不同，常见为数分钟到二三十分钟）。
- 运行时可通过日志判断是否在继续执行（会输出读取、聚合、写文件等步骤日志）。
- 如果长期无输出，可先检查磁盘 I/O、内存占用，以及是否正在首次构建 GeoJSON。

### 7) 关键假设与限制
- 面积/长度相关计算统一以 **EPSG:2263** 为基准，脚本会进行 CRS 校验与必要重投影。
- 高度采用代理：`楼层数 × 10 ft`，不是 LiDAR/三维真高。
- `building_density` 当前为 proxy（`GFA/楼层` 估计 footprint），适合宏观比较，不等价于精确建筑覆盖率。
- 热力图使用 `quantiles (k=11)`，并对每个指标在 `99% 分位`做上限截断，避免极端值主导颜色。
- 若 MapPLUTO 目标列缺失，脚本会记录 warning，并对缺失数值列按 0 或 NaN 处理后继续。
- 热力图量化分级依赖 `mapclassify`；若不可用会回退到连续色带绘图并仍保存 PNG。

### 8) 交付给组员/整合人的字段清单
`data/processed/block_features.csv` 主要包含（可能含几何来源附加列）：
- 键与几何相关：`block_id`, `Borough`, `Block`, `block_area_ft2`
- parcel 聚合：`bldgarea_sum`, `resarea_sum`, `numfloors_mean`, `numfloors_std`, `builtfar_mean`, `parcel_count`
- 核心输出：`footprint_area_sum`, `gross_floor_area_sum`, `building_density`, `FAR`, `height_mean`, `height_std`, `sky_openness_proxy`

`block_id` 规则：
- `block_id = Borough + "_" + Block`（示例：`MN_123`）



### Week X – Dynamic Feature Alignment（本周新增）
- 新增脚本：
  - `python -m src.compute_dynamic_features --config config.yaml` → `data/processed/exposure_weekly.csv`
  - `python -m src.build_nsi_input --config config.yaml` → `data/processed/nsi_input_weekly.csv`
- 动态特征按 `bgrp_id × week_start`（周一）对齐，缺失 exposure 在 merge 后填 0，并校验键唯一性。
- 本周不改动上周 block-level 产线逻辑，仅新增动态对齐模块。

### 9) Troubleshooting
- **依赖安装失败**  
  - 建议先升级 pip，再安装：
    ```powershell
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
    ```
- **`geopandas` / `fiona` / `pyproj` 安装问题**  
  - 常见于 GDAL 相关二进制依赖；优先使用 Conda 环境或与系统匹配的 wheels。
- **`mapclassify` 缺失导致 quantile 失败**  
  - 已写入 `requirements.txt`；缺失时脚本会给出清晰报错并自动回退到连续色带。
- **GDB 读取报错（layer 找不到）**  
  - 检查 `config.yaml` 中 `mappluto_gdb` 与 `mappluto_layer` 是否正确。
- **运行很慢**  
  - MapPLUTO 文件较大，先确认进程仍在运行并查看日志；避免把数据放在慢速网络盘。
- **输出出现空值或几乎全 0**  
  - 检查 `Borough/Block` 是否成功识别；确认 `blocks.geojson` 的 `block_id` 与 parcel 聚合键一致；查看日志中的 join hit rate。

---

## English Documentation (for teammates and PI)


### 0) Scope of this documentation update
- Per request, this collaboration updates **README.md only** (no changes to `src/`, `config.example.yaml`, `requirements.txt`, or data files).
- The goal is to make the pipeline runnable end-to-end directly from this README.

### 1) Project Overview
This repository supports **Project_Urban_Noise (SYSEN 5900)** by transforming NYC urban geospatial data (primarily MapPLUTO) into **block-level urban form features** for downstream integration and modeling, delivered to **Zijie Xiong + Yingyi Shu**. The implemented pipeline is: read MapPLUTO parcel/lot data → dissolve to block geometries (`blocks.geojson`) → compute block-level features (`CSV/GeoJSON`) → generate red-high quantile heatmaps with 99th-percentile clipping.

### 2) Data Inputs
> Keep raw inputs local under `data/raw/` and do not commit them.

Core input used by current stable pipeline:
- `data/raw/MapPLUTO25v4_unclipped.gdb` (layer: `MapPLUTO_25v4_unclipped`)

Optional inputs (not required for current core metrics):
- `data/raw/BUILDING_20260209.csv`
- `data/raw/bldg_3d_metrics.csv`
- `data/raw/Centerline_20260209.csv`

Runtime config:
- `config.yaml` (copy from `config.example.yaml`)

### 3) Outputs
The pipeline produces:
- `data/processed/blocks.geojson`
- `data/processed/block_features.geojson`
- `data/processed/block_features.csv`
- `figures/heatmap_building_density_quantiles_red.png`
- `figures/heatmap_height_heterogeneity_quantiles_red.png`

### 4) Metric Definitions (block-level)
Let block area be `block_area_ft2`, and parcel aggregation provide `bldgarea_sum` (gross floor area proxy) and `numfloors_mean`.

- `building_density` (coverage/density proxy)
  - `footprint_area_sum = bldgarea_sum / max(numfloors_mean, 1)`
  - `building_density = footprint_area_sum / block_area_ft2`
- `FAR`
  - `FAR = gross_floor_area_sum / block_area_ft2`, where `gross_floor_area_sum = bldgarea_sum`
- `height_mean`, `height_std` (height heterogeneity)
  - `height_mean = numfloors_mean * 10 ft`
  - `height_std = numfloors_std * 10 ft`
- `sky_openness_proxy`
  - `sky_openness_proxy = clip(1 - building_density, 0, 1)`
- `street_hw_ratio`
  - Not included in the current stable output (optional / future extension with centerline-width-based or proxy computation).

### 5) How to Run (in order)
Run from repository root.

1. Prepare config:
```powershell
Copy-Item config.example.yaml config.yaml
```

2. Build block geometries:
```powershell
python -m src.build_blocks --config config.yaml
```

3. Compute block features:
```powershell
python -m src.compute_block_features --config config.yaml
```

4. Generate heatmaps (red=high, quantiles, 99% clipping):
```powershell
python -m src.plot_heatmaps --config config.yaml
```

### 6) Performance Notes
- Reading/dissolving the MapPLUTO GDB can be slow (often several minutes, and up to ~20–30 minutes depending on machine/storage).
- Watch logs to confirm progress (read/aggregate/write steps are logged).
- If output stalls, check disk I/O, memory, and whether this is the first large GeoJSON write.

### 7) Key Assumptions and Limitations
- Area/length-sensitive operations are standardized to **EPSG:2263** (validated/reprojected by scripts).
- Height is proxied by `floors × 10 ft`, not true 3D/LiDAR building height.
- `building_density` is currently a proxy (`GFA/floors` footprint estimate), suitable for comparative analysis rather than exact coverage.
- Heatmaps use `quantiles (k=11)` with per-metric 99th-percentile clipping to reduce outlier dominance.
- If target MapPLUTO columns are missing, warnings are logged and missing numeric fields are handled as 0/NaN where applicable.
- Quantile classification depends on `mapclassify`; if unavailable, plotting falls back to continuous scale and still saves PNGs.

### 8) Delivered Field List (for integration)
`data/processed/block_features.csv` includes (plus possible geometry-source carry-over fields):
- Keys/geometry-related: `block_id`, `Borough`, `Block`, `block_area_ft2`
- Parcel aggregates: `bldgarea_sum`, `resarea_sum`, `numfloors_mean`, `numfloors_std`, `builtfar_mean`, `parcel_count`
- Core outputs: `footprint_area_sum`, `gross_floor_area_sum`, `building_density`, `FAR`, `height_mean`, `height_std`, `sky_openness_proxy`

`block_id` rule:
- `block_id = Borough + "_" + Block` (example: `MN_123`)

### 9) Troubleshooting
- **Dependency installation errors**
  ```powershell
  python -m pip install --upgrade pip
  python -m pip install -r requirements.txt
  ```
- **`geopandas` / `fiona` / `pyproj` install issues**
  - Usually related to GDAL binary dependencies; prefer a Conda environment or matching wheels.
- **Missing `mapclassify` for quantiles**
  - It is listed in `requirements.txt`; script logs a clear message and falls back to continuous plotting.
- **GDB read/layer errors**
  - Verify `mappluto_gdb` and `mappluto_layer` in `config.yaml`.
- **Pipeline is slow**
  - Large MapPLUTO input is expected; confirm process is alive and logs are updating.
- **Many null/zero outputs**
  - Check Borough/Block detection, `block_id` consistency between `blocks.geojson` and parcel aggregation, and inspect join hit rate logs.



## Week X – Dynamic Feature Alignment

This week adds a **new dynamic alignment layer** at `bgrp_id × week_start` for residual modeling, while keeping last week’s block-feature pipeline unchanged.

### New scripts
1. `python -m src.compute_dynamic_features --config config.yaml`
   - Output: `data/processed/exposure_weekly.csv`
   - Required columns: `bgrp_id`, `week_start`, `traffic`, `street_activity`, `dob_permits`

2. `python -m src.build_nsi_input --config config.yaml`
   - Output: `data/processed/nsi_input_weekly.csv`
   - Required columns: `bgrp_id`, `week_start`, `y_total`, `households`, `traffic`, `street_activity`, `dob_permits`

### Alignment rules implemented
- Spatial base comes from `data/processed/blocks.geojson`, with CRS normalized to EPSG:2263 before spatial operations.
- `week_start` is Monday-based (`YYYY-MM-DD`).
- Robust column detection and logged fallback logic are used (no fragile single-name dependencies).
- Missing exposures are filled with `0` after merge.
- Key uniqueness on `(bgrp_id, week_start)` is enforced.
- Quality summary logs include number of bgrps, number of weeks, and min/max diagnostics.

### Notes on fallbacks
- Traffic: if direct bgrp alignment is unavailable, the script applies key/proxy fallback and logs warnings.
- Street activity: Active Cabaret/Catering licenses are treated as a static proxy and replicated weekly.
- Construction: DOB permits are weekly if date fields exist; otherwise a documented fallback is used and logged.
