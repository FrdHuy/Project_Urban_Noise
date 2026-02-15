import requests
import time
import sys
import csv
import io
import random

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# NYC 311 数据集 CSV 接口
BASE = "https://data.cityofnewyork.us/resource/erm2-nwe9.csv"
DEFAULT_BASENAME = "nyc_311_noise"

# 批量参数设置
BATCH = 50000        # 每批最大行数（Socrata API 限制：最多 50,000）
TIMEOUT = 120        # 网络请求超时时间（秒）
SLEEP_SEC = 2        # 每批之间暂停，避免触发限流
RETRIES = 5          # 每次请求的最大重试次数
BACKOFF_SEC = 3      # 初始退避秒数（指数退避）

# 选择的字段（只保留和噪音分析相关的）
SELECT_COLS = [
    "unique_key",          # 每条工单唯一ID
    "created_date",        # 工单创建时间
    "closed_date",         # 工单关闭时间
    "status",              # 工单状态（Open/Closed等）
    "complaint_type",      # 投诉大类（Noise - Residential等）
    "descriptor",          # 投诉细类（Party, Vehicle, Construction等）
    "incident_zip",        # 邮编
    "latitude",            # 纬度
    "longitude",           # 经度
    "location"             # 经纬度组合（GIS工具可直接用）
]
SELECT = ",".join(SELECT_COLS)

def build_where_for_year(year):
    return (
        f"created_date >= '{year}-01-01T00:00:00' AND "
        f"created_date <= '{year}-12-31T23:59:59' AND "
        "complaint_type like 'Noise%'"
    )

# 请求函数：按 offset 分页抓取数据
def fetch_batch(offset, where_clause, limit=BATCH):
    params = {
        "$select": SELECT,   # 只取指定字段
        "$where": where_clause,     # 筛选条件
        "$order": ":id",     # 保证分页稳定，不重复/漏数据
        "$limit": limit,     # 每批行数
        "$offset": offset    # 起始位置
    }

    # 分别设置连接/读取超时，配合重试与指数退避
    connect_timeout = 20
    read_timeout = max(60, TIMEOUT)  # 读取可能较慢，适当放宽

    last_exc = None
    for attempt in range(1, RETRIES + 1):
        try:
            r = requests.get(BASE, params=params, timeout=(connect_timeout, read_timeout))
            r.raise_for_status()
            return r.content
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            last_exc = e
            if attempt == RETRIES:
                break
            # 指数退避 + 抖动
            sleep_s = BACKOFF_SEC * (2 ** (attempt - 1)) + random.uniform(0, 0.5)
            print(f"请求超时/连接异常，重试第 {attempt}/{RETRIES} 次后等待 {sleep_s:.1f}s …")
            time.sleep(sleep_s)
        except requests.exceptions.RequestException as e:
            # 非超时类错误直接抛出
            raise
    # 若所有重试仍失败，抛出最后的异常
    raise last_exc if last_exc else RuntimeError("未知错误：请求失败")

# 写入第一批：含表头
def write_first_batch(path, content_bytes):
    with open(path, "wb") as f:
        f.write(content_bytes)

# 追加后续批次：去掉表头再写入
def append_without_header(path, content_bytes):
    nl = content_bytes.find(b"\n")  # 找到首行（表头）的换行符位置
    if nl == -1:
        return 0
    body = content_bytes[nl+1:]     # 跳过表头部分
    with open(path, "ab") as f:     # 以追加模式写入
        f.write(body)
    return len(body)

# 清洗与序列化：
# - 若 location 为空且有 latitude/longitude，则生成 location
# - 若 latitude 与 longitude 都为空，则丢弃该行
# - include_header=True 时输出包含表头；否则仅输出数据行
def transform_and_serialize(content_bytes, include_header=True):
    text = content_bytes.decode("utf-8", errors="replace")
    input_io = io.StringIO(text)

    reader = csv.DictReader(input_io)
    fieldnames = reader.fieldnames or []

    # 兼容字段名大小写或缺失
    def pick(name):
        if name in fieldnames:
            return name
        lower_map = {fn.lower(): fn for fn in fieldnames}
        return lower_map.get(name.lower(), name)

    latitude_key = pick("latitude")
    longitude_key = pick("longitude")
    location_key = pick("location")

    output_io = io.StringIO()
    writer = csv.DictWriter(output_io, fieldnames=fieldnames, lineterminator="\n")
    if include_header:
        writer.writeheader()

    filled_location_count = 0
    dropped_count = 0
    written_rows = 0

    for row in reader:
        lat = (row.get(latitude_key) or "").strip()
        lon = (row.get(longitude_key) or "").strip()
        loc = (row.get(location_key) or "").strip()

        # 两个经纬度都缺失则丢弃
        if not lat and not lon:
            dropped_count += 1
            continue

        # 只有 location 缺失且有经纬度则补齐
        if not loc and lat and lon:
            # 使用 (lat, lon) 形式，便于直观检查；需要其他格式可再调整
            row[location_key] = f"({lat}, {lon})"
            filled_location_count += 1

        writer.writerow(row)
        written_rows += 1

    output_bytes = output_io.getvalue().encode("utf-8", errors="replace")
    return output_bytes, filled_location_count, dropped_count, written_rows

def main():
    for year in range(2022, 2025):
        out_file = f"{DEFAULT_BASENAME}_{year}.csv"
        where_clause = build_where_for_year(year)

        offset = 0
        batch_idx = 0
        total_bytes = 0
        total_filled_location = 0
        total_dropped = 0

        print(f"开始抓取 {year} 年数据 -> {out_file}")

        # 循环抓取直到没有数据返回
        while True:
            blob = fetch_batch(offset, where_clause)
            size = len(blob)

            # 如果返回内容太小，说明数据抓取结束
            if size < 200:
                print(f"{year} 年抓取完成")
                break

            # 清洗 + 序列化
            include_header = (batch_idx == 0)
            out_bytes, filled_cnt, dropped_cnt, written_rows = transform_and_serialize(
                blob, include_header=include_header
            )

            # 写入文件
            if batch_idx == 0:
                write_first_batch(out_file, out_bytes)
                total_bytes += len(out_bytes)
                print(f"[批 {batch_idx+1}] 写入 {len(out_bytes)} bytes（含表头，清洗后 {written_rows} 行）")
            else:
                with open(out_file, "ab") as f:
                    f.write(out_bytes)
                total_bytes += len(out_bytes)
                print(f"[批 {batch_idx+1}] 追加 {len(out_bytes)} bytes（去表头，清洗后 {written_rows} 行）")

            total_filled_location += filled_cnt
            total_dropped += dropped_cnt

            # 下一批参数更新
            batch_idx += 1
            offset += BATCH
            time.sleep(SLEEP_SEC)

        print(f"完成：{out_file}（累计 ~{total_bytes/1024/1024:.2f} MB）")
        print(f"补全 location 条数：{total_filled_location}")
        print(f"删除无经纬度数据条数：{total_dropped}")

if __name__ == "__main__":
    main()
