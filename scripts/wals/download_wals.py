#!/usr/bin/env python3
"""
从GitHub下载WALS数据集（CLDF格式）
数据来源：https://github.com/cldf-datasets/wals
"""

import requests
import pandas as pd
import os
from pathlib import Path
import time

# 配置路径
OUTPUT_DIR = Path("data/external_wals")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# GitHub仓库文件URL
BASE_URL = "https://raw.githubusercontent.com/cldf-datasets/wals/master/cldf/"

# 需要下载的文件列表
FILES = [
    "languages.csv",
    "values.csv",
    "parameters.csv",
    "codes.csv",
]

def download_file(filename, retries=3):
    """下载单个CSV文件"""
    url = BASE_URL + filename
    output_path = OUTPUT_DIR / filename

    print(f"下载 {filename}...")

    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()

            # 保存文件
            with open(output_path, 'wb') as f:
                f.write(response.content)

            print(f"  保存到 {output_path} ({len(response.content)} 字节)")
            return True

        except Exception as e:
            print(f"  尝试 {attempt + 1}/{retries} 失败: {e}")
            if attempt < retries - 1:
                time.sleep(2)

    return False

def validate_files():
    """验证下载的文件"""
    print("\n验证文件...")

    for filename in FILES:
        filepath = OUTPUT_DIR / filename
        if not filepath.exists():
            print(f"  ❌ {filename}: 文件不存在")
            return False

        # 尝试读取CSV验证格式
        try:
            df = pd.read_csv(filepath)
            print(f"  ✓ {filename}: {len(df)} 行, {len(df.columns)} 列")
        except Exception as e:
            print(f"  ❌ {filename}: CSV解析错误 - {e}")
            return False

    return True

def main():
    print("开始下载WALS数据集...")
    print(f"GitHub仓库: {BASE_URL}")
    print(f"输出目录: {OUTPUT_DIR.absolute()}\n")

    # 下载所有文件
    success_count = 0
    for filename in FILES:
        if download_file(filename):
            success_count += 1

    print(f"\n下载完成: {success_count}/{len(FILES)} 个文件")

    # 验证文件
    if success_count == len(FILES):
        if validate_files():
            print("\n✅ WALS数据集下载完成!")

            # 显示数据概览
            print("\n数据概览:")
            languages_path = OUTPUT_DIR / "languages.csv"
            params_path = OUTPUT_DIR / "parameters.csv"
            values_path = OUTPUT_DIR / "values.csv"

            langs_df = pd.read_csv(languages_path)
            params_df = pd.read_csv(params_path)
            values_df = pd.read_csv(values_path)

            print(f"  - 语言数量: {len(langs_df)}")
            print(f"  - 特征参数数量: {len(params_df)}")
            print(f"  - 特征值数量: {len(values_df)}")

            # 显示前几个特征ID
            print(f"  - 特征示例: {list(params_df['ID'].head(5))}")

        else:
            print("\n⚠️ 文件验证失败")
    else:
        print("\n❌ 下载失败，请检查网络连接")

if __name__ == "__main__":
    main()