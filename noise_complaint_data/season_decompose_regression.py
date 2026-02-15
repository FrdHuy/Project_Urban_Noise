##############################################
# NYC 噪音投诉：季节分解 + 回归组合模型（Python 版）
#---------------------------------------------
# 步骤：
# 1. 读入原始逐条投诉数据
# 2. 按月聚合成时间序列
# 3. STL 分解：Trend + Seasonality + Remainder
# 4. 对 Trend 做回归（时间趋势）
# 5. 对 Seasonality 做月份 dummy 回归
# 6. 组合得到未来 12 个月预测
##############################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL
import statsmodels.api as sm
import os

# =========================
# 1. 读入数据
# =========================

# 获取脚本所在目录
# 处理IDE中__file__可能不存在的情况
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    # 如果__file__不存在，使用当前工作目录
    script_dir = os.getcwd()
    print(f"Warning: __file__ not available, using current working directory: {script_dir}")

# 切换到脚本所在目录（解决IDE工作目录问题）
original_cwd = os.getcwd()
try:
    os.chdir(script_dir)
    print(f"Changed working directory to: {script_dir}")
except Exception as e:
    print(f"Warning: Could not change directory: {e}")

input_file = os.path.join(script_dir, "noise_311.csv")

# 检查文件是否存在
if not os.path.exists(input_file):
    # 尝试在当前工作目录查找
    alt_file = os.path.join(os.getcwd(), "noise_311.csv")
    if os.path.exists(alt_file):
        input_file = alt_file
        script_dir = os.getcwd()
        print(f"Found input file in current working directory: {input_file}")
    else:
        error_msg = f"Input file not found.\n"
        error_msg += f"  Searched in script directory: {os.path.join(script_dir, 'noise_311.csv')}\n"
        error_msg += f"  Searched in current directory: {alt_file}\n"
        error_msg += f"  Current working directory: {os.getcwd()}\n"
        error_msg += f"  Script directory: {script_dir}"
        raise FileNotFoundError(error_msg)

# 假设 CSV 文件名为 noise_311.csv，包含 created_date 列
df_raw = pd.read_csv(input_file)

# 根据实际列名修改这里
df_raw["created_date"] = pd.to_datetime(df_raw["created_date"])

# =========================
# 2. 按月聚合成时间序列
# =========================

df_raw["month"] = df_raw["created_date"].dt.to_period("M").dt.to_timestamp()
df_month = (
    df_raw.groupby("month")
    .size()
    .reset_index(name="y")
    .sort_values("month")
)

print(df_month.head())
print(df_month.tail())

# =========================
# 3. STL 分解
# =========================

y = df_month["y"].values
stl = STL(y, period=12)
res = stl.fit()

trend = res.trend
seasonal = res.seasonal
remainder = res.resid

# 检查并处理NaN值（STL分解在边界可能产生NaN）
print(f"\nSTL Decomposition Summary:")
print(f"Trend: {np.sum(np.isnan(trend))} NaN values")
print(f"Seasonal: {np.sum(np.isnan(seasonal))} NaN values")
print(f"Remainder: {np.sum(np.isnan(remainder))} NaN values")

# 移除NaN值对应的行
valid_mask = ~(np.isnan(trend) | np.isnan(seasonal) | np.isnan(remainder))
df_month = df_month[valid_mask].reset_index(drop=True)
trend = trend[valid_mask]
seasonal = seasonal[valid_mask]
remainder = remainder[valid_mask]

print(f"After removing NaN: {len(df_month)} valid observations")

# STL 分解图
fig = res.plot()
fig.set_size_inches(10, 8)
plt.tight_layout()
stl_plot_path = os.path.join(script_dir, "stl_decomposition_py.png")
plt.savefig(stl_plot_path, dpi=150)
plt.close()

# =========================
# 4. 趋势回归（Trend ~ t）
# =========================

t_index = np.arange(1, len(trend) + 1)

df_trend = pd.DataFrame({
    "t": t_index,
    "trend": trend
})

X_trend = sm.add_constant(df_trend["t"])
trend_model = sm.OLS(df_trend["trend"], X_trend).fit()
print(trend_model.summary())

df_trend["trend_fit"] = trend_model.predict(X_trend)

plt.figure(figsize=(10, 5))
plt.plot(df_trend["t"], df_trend["trend"], label="Trend", alpha=0.7)
plt.plot(df_trend["t"], df_trend["trend_fit"], label="Fitted Trend", linestyle="--")
plt.title("Trend and Fitted Trend")
plt.xlabel("Time index")
plt.ylabel("Trend level")
plt.legend()
plt.tight_layout()
trend_plot_path = os.path.join(script_dir, "trend_fit_py.png")
plt.savefig(trend_plot_path, dpi=150)
plt.close()

# =========================
# 5. 季节性回归（seasonal ~ month_dummy）
# =========================
# 注意：应该对STL分解得到的seasonal成分进行回归，而不是原始数据y

df_season = pd.DataFrame({
    "seasonal": seasonal,  # 使用STL分解的seasonal成分
    "month_num": df_month["month"].dt.month
})

# 确保seasonal是数值类型
df_season["seasonal"] = pd.to_numeric(df_season["seasonal"], errors='coerce')

# 生成 12 个月 dummy，其中一个会被自动当作基准
month_dummies = pd.get_dummies(df_season["month_num"], prefix="m", drop_first=True)
X_season = sm.add_constant(month_dummies)

# 转换为numpy数组以确保数据类型正确
seasonal_values = np.asarray(df_season["seasonal"], dtype=float)
# 确保X_season也是数值类型
X_season = np.asarray(X_season, dtype=float)

season_model = sm.OLS(seasonal_values, X_season).fit()
print(season_model.summary())

df_season["season_fit"] = season_model.predict(X_season)

plt.figure(figsize=(8, 5))
plt.scatter(df_season["month_num"], df_season["seasonal"], alpha=0.4, label="STL Seasonal Component")
plt.scatter(df_season["month_num"], df_season["season_fit"], color="red", label="Season Fit")
plt.xticks(range(1, 13))
plt.xlabel("Month (1-12)")
plt.ylabel("Seasonal component")
plt.title("Seasonality Fit by Month (STL Seasonal Component)")
plt.legend()
plt.tight_layout()
season_plot_path = os.path.join(script_dir, "season_fit_py.png")
plt.savefig(season_plot_path, dpi=150)
plt.close()

# =========================
# 6. 组合预测（未来 12 个月）
# =========================

h = 12  # 预测未来一年
last_t = df_trend["t"].iloc[-1]
future_t = np.arange(last_t + 1, last_t + h + 1)

# 6.1 趋势预测
X_future_trend = sm.add_constant(pd.Series(future_t, name="t"))
trend_pred = trend_model.predict(X_future_trend)

# 6.2 季节性预测
last_month = df_month["month"].iloc[-1]
future_month_seq = pd.date_range(
    start=(last_month + pd.offsets.MonthBegin(1)),
    periods=h,
    freq="MS"
)

future_month_num = future_month_seq.month
future_month_dummies = pd.get_dummies(future_month_num, prefix="m", drop_first=True)

# 确保 dummy 列与训练阶段一致
future_month_dummies = future_month_dummies.reindex(columns=month_dummies.columns, fill_value=0)
X_future_season = sm.add_constant(future_month_dummies)
season_pred = season_model.predict(X_future_season)

# 6.3 组合预测
y_pred = trend_pred + season_pred

df_forecast = pd.DataFrame({
    "month": future_month_seq,
    "trend_pred": trend_pred,
    "season_pred": season_pred,
    "y_pred": y_pred
})

print(df_forecast)

# =========================
# 7. 历史 + 预测可视化
# =========================

df_hist_plot = df_month[["month", "y"]].copy()
df_hist_plot["type"] = "Historical"

df_forecast_plot = df_forecast[["month", "y_pred"]].rename(columns={"y_pred": "y"})
df_forecast_plot["type"] = "Forecast"

df_plot = pd.concat([df_hist_plot, df_forecast_plot], ignore_index=True)

plt.figure(figsize=(12, 6))
for label, group in df_plot.groupby("type"):
    plt.plot(group["month"], group["y"], marker="o", label=label)

plt.title("Monthly Noise Complaints: History & 12-Month Forecast")
plt.xlabel("Month")
plt.ylabel("Complaint count")
plt.legend()
plt.tight_layout()
forecast_plot_path = os.path.join(script_dir, "history_forecast_py.png")
plt.savefig(forecast_plot_path, dpi=150)
plt.close()

print("\n" + "="*60)
print("Analysis completed successfully!")
print("="*60)
print(f"Output directory: {script_dir}")
print(f"Generated plots:")
print(f"  - {stl_plot_path}")
print(f"  - {trend_plot_path}")
print(f"  - {season_plot_path}")
print(f"  - {forecast_plot_path}")
print("="*60)
##############################################
# 生成的图片：
# - stl_decomposition_py.png
# - trend_fit_py.png
# - season_fit_py.png
# - history_forecast_py.png
# 都可以直接放在你的报告 / PPT 中。
##############################################
