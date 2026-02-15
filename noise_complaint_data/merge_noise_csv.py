import pandas as pd
import glob
import re
import os

def merge_noise_files():
    # 获取脚本所在的目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    print("Script directory:", script_dir)
    print("Current working directory:", os.getcwd())

    # 在脚本所在目录中匹配 .csv 文件
    csv_pattern = os.path.join(script_dir, "nyc_311_noise_*.csv")
    CSV_pattern = os.path.join(script_dir, "nyc_311_noise_*.CSV")
    files = glob.glob(csv_pattern) + glob.glob(CSV_pattern)
    
    # 转换为绝对路径并去重（使用dict.fromkeys保持顺序）
    files = [os.path.abspath(f) for f in files]
    files = list(dict.fromkeys(files))  # 去重但保持顺序
    
    # 排除输出文件本身（如果已存在），避免重复读取
    output_name = os.path.join(script_dir, "noise_311.csv")
    files = [f for f in files if f != os.path.abspath(output_name)]

    print("\nMatched files:")
    print(files)

    if not files:
        print("No matching files found.")
        print("Please check that files are named like nyc_311_noise_2016.csv")
        print("and that this script is in the same folder.")
        return

    # 按年份排序（例如 2016 -> 2024）
    def extract_year(filename):
        match = re.search(r"(\d{4})", filename)
        return int(match.group(1)) if match else 9999

    files = sorted(files, key=extract_year)

    print("\nFiles in order:")
    for f in files:
        print(" -", f)

    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f, low_memory=False)
            print("Read", f, "successfully with shape:", df.shape)
            dfs.append(df)
        except UnicodeDecodeError:
            print("Encoding issue when reading", f, "trying latin1...")
            df = pd.read_csv(f, encoding="latin1", low_memory=False)
            print("Read", f, "successfully with shape:", df.shape)
            dfs.append(df)
        except Exception as e:
            print("Error reading", f, ":", e)

    if not dfs:
        print("\nNo DataFrames were loaded, cannot concatenate.")
        return

    # 合并所有数据框
    df_all = pd.concat(dfs, ignore_index=True)
    
    # 去重（如果有完全重复的行）
    original_shape = df_all.shape
    df_all = df_all.drop_duplicates()
    if df_all.shape[0] < original_shape[0]:
        print(f"\nRemoved {original_shape[0] - df_all.shape[0]} duplicate rows.")

    # 使用绝对路径保存输出文件
    df_all.to_csv(output_name, index=False)

    print("\n" + "="*60)
    print("Merge completed successfully!")
    print("="*60)
    print(f"Output file path: {output_name}")
    print(f"Output directory: {script_dir}")
    print(f"Final data shape: {df_all.shape[0]} rows and {df_all.shape[1]} columns")
    print("="*60)

if __name__ == "__main__":
    merge_noise_files()
