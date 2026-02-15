import pandas as pd
import numpy as np
import statsmodels.api as sm
import os

# =========================
# 0. 基本设置
# =========================

# 脚本所在目录（保证用 VSCode 运行也能找到文件）
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    script_dir = os.getcwd()

os.chdir(script_dir)
print("Working directory:", script_dir)

# =========================
# 1. 读取噪音数据 noise_311.csv
# =========================

noise = pd.read_csv("noise_311.csv")

# 把 created_date 转成 datetime
noise["created_date"] = pd.to_datetime(noise["created_date"], errors="coerce")

# 只保留有日期的行
noise = noise.dropna(subset=["created_date"])

# 提取年份
noise["year"] = noise["created_date"].dt.year

# 只保留 2016–2024
noise = noise[(noise["year"] >= 2016) & (noise["year"] <= 2024)]

# 按年计数：每条记录就是 1 起噪音投诉
y_year = (
    noise.groupby("year")
    .size()
    .reset_index(name="noise_complaints")
    .sort_values("year")
)

print("\nAnnual noise complaints (from noise_311.csv):")
print(y_year)

# =========================
# 2. 读取 DataSet.csv 并构造年度自变量
# =========================

df = pd.read_csv("DataSet.csv")

# 我们要用的列前缀：
#   Housing_Complaint_Count_YYYY
#   Environment_Complaint_Count_YYYY
#   PublicService_Complaint_Count_YYYY
#   DOB_Count_YYYY
years = list(range(2015, 2026))  # DataSet 里从 2015 开始到 2025
rows = []

for year in years:
    row = {"year": year}

    # 对每个大类求和（跨所有 block group 求和）
    col_housing = f"Housing_Complaint_Count_{year}"
    col_env = f"Environment_Complaint_Count_{year}"
    col_pub = f"PublicService_Complaint_Count_{year}"
    col_dob = f"DOB_Count_{year}"

    if col_housing in df.columns:
        row["housing_total"] = df[col_housing].sum(skipna=True)
    else:
        row["housing_total"] = np.nan

    if col_env in df.columns:
        row["environment_total"] = df[col_env].sum(skipna=True)
    else:
        row["environment_total"] = np.nan

    if col_pub in df.columns:
        row["publicservice_total"] = df[col_pub].sum(skipna=True)
    else:
        row["publicservice_total"] = np.nan

    if col_dob in df.columns:
        row["dob_total"] = df[col_dob].sum(skipna=True)
    else:
        row["dob_total"] = np.nan

    rows.append(row)

exog_year = pd.DataFrame(rows)
exog_year = exog_year.sort_values("year")

print("\nAnnual exogenous variables summary (from DataSet.csv):")
print(exog_year.head(12))

# =========================
# 3. 合并成年度时序表 2016–2024
# =========================

df_ts = (
    y_year.merge(exog_year, on="year", how="left")
          .sort_values("year")
          .reset_index(drop=True)
)

# 只保留 2016–2024，这段时间既有噪音投诉也有 exog
df_ts = df_ts[(df_ts["year"] >= 2016) & (df_ts["year"] <= 2024)].reset_index(drop=True)

print("\nMerged annual time series (2016–2024):")
print(df_ts)

# 如果有缺失，可以看一下（看 DataSet 有没有对应年份）
print("\nMissing values per column:")
print(df_ts.isna().sum())

# 简单地把有 NaN 的年份删掉（也可以选择插值）
df_ts = df_ts.dropna().reset_index(drop=True)

# =========================
# 4. 构造时间趋势 t，自变量矩阵 X，因变量 y
# =========================

# 时间索引 t：2016->1, 2017->2, ...
df_ts["t"] = np.arange(1, len(df_ts) + 1)

# 选择自变量列：可以根据需要调整
exog_cols = ["t", "housing_total", "environment_total", "publicservice_total", "dob_total"]

X = df_ts[exog_cols]
X = sm.add_constant(X)       # 截距项
y = df_ts["noise_complaints"].values

print("\nDesign matrix columns:", X.columns.tolist())
print("Number of observations:", len(df_ts))

# =========================
# 5. 拟合 OLS 时序回归模型
# =========================

model = sm.OLS(y, X).fit()
print("\n================ OLS Annual Time-series Regression (with exogenous variables) ================")
print(model.summary())

# 拟合值
df_ts["y_hat"] = model.predict(X)

# =========================
# 6. 简单画一下真实值 vs 拟合值
# =========================
try:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 5))
    plt.plot(df_ts["year"], df_ts["noise_complaints"], marker="o", label="Observed")
    plt.plot(df_ts["year"], df_ts["y_hat"], marker="s", linestyle="--", label="Fitted")
    plt.xlabel("Year")
    plt.ylabel("Noise complaints (total)")
    plt.title("Annual noise complaints: observed vs fitted\n(Trend + exogenous variables)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("annual_ts_regression_fit.png", dpi=150)
    plt.close()
    print("\nSaved plot: annual_ts_regression_fit.png")
except Exception as e:
    print("\nPlotting failed but model fitted successfully:", e)

print("\nDone. df_ts with fitted values:")
print(df_ts)
