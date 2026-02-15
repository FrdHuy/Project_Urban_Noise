#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试脚本：检查所有必需的库是否正确安装
"""

import sys
import os

print("="*60)
print("Python Environment Check")
print("="*60)
print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")
print(f"Current working directory: {os.getcwd()}")

# 检查 __file__ 是否可用
try:
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)
    print(f"Script file: {script_path}")
    print(f"Script directory: {script_dir}")
    print(f"__file__ available: Yes")
except NameError:
    script_dir = os.getcwd()
    print(f"__file__ available: No (using current directory)")
    print(f"Script directory (fallback): {script_dir}")

# 检查关键文件是否存在
print("\nChecking for required files:")
noise_file = os.path.join(script_dir, "noise_311.csv")
print(f"  noise_311.csv in script dir: {os.path.exists(noise_file)}")
if os.path.exists(noise_file):
    print(f"    Path: {noise_file}")
else:
    # 检查当前目录
    alt_file = os.path.join(os.getcwd(), "noise_311.csv")
    print(f"  noise_311.csv in current dir: {os.path.exists(alt_file)}")
    if os.path.exists(alt_file):
        print(f"    Path: {alt_file}")

print("="*60)

# 测试导入
print("\nTesting imports...")
try:
    import pandas as pd
    print(f"[OK] pandas {pd.__version__}")
except ImportError as e:
    print(f"[FAIL] pandas: {e}")

try:
    import numpy as np
    print(f"[OK] numpy {np.__version__}")
except ImportError as e:
    print(f"[FAIL] numpy: {e}")

try:
    import matplotlib
    print(f"[OK] matplotlib {matplotlib.__version__}")
except ImportError as e:
    print(f"[FAIL] matplotlib: {e}")

try:
    import statsmodels
    print(f"[OK] statsmodels {statsmodels.__version__}")
except ImportError as e:
    print(f"[FAIL] statsmodels: {e}")

try:
    from statsmodels.tsa.seasonal import STL
    print("[OK] statsmodels.tsa.seasonal.STL")
except ImportError as e:
    print(f"[FAIL] statsmodels.tsa.seasonal.STL: {e}")

try:
    import statsmodels.api as sm
    print("[OK] statsmodels.api")
except ImportError as e:
    print(f"[FAIL] statsmodels.api: {e}")

print("\n" + "="*60)
print("Import test completed!")
print("="*60)

