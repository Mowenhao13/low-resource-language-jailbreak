#!/usr/bin/env python3
"""
从WALS公开数据库获取12种语言的特征数据。

功能：
1. 从GitHub下载WALS数据集（CLDF格式）
2. 提取指定语言的特征值
3. 为每种语言生成CSV文件，包含Fid、Value、Feature三列

目标语言：zu, gd, hmn, gn, uk, bn, th, he, zh-CN, ar, it, hi

输出文件：data/wals/{语言代码}_wals.csv

依赖：requests, pandas
"""

import requests
import pandas as pd
import os
import sys
from pathlib import Path
import time
import re

# 配置路径
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
EXTERNAL_WALS_DIR = DATA_DIR / "external_wals"
WALS_OUTPUT_DIR = DATA_DIR / "wals"

# 确保目录存在
EXTERNAL_WALS_DIR.mkdir(parents=True, exist_ok=True)
WALS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# GitHub仓库文件URL
BASE_URL = "https://raw.githubusercontent.com/cldf-datasets/wals/master/cldf/"

# 需要下载的文件列表
FILES = [
    "languages.csv",
    "values.csv",
    "parameters.csv",
    "codes.csv",
]

# 语言代码映射：用户提供的代码 -> ISO 639-3代码（用于WALS查找）
LANGUAGE_MAPPING = {
    "zu": "zul",      # Zulu
    "gd": "gla",      # Scottish Gaelic
    "hmn": "hnj",     # Hmong Njua
    "gn": "gug",      # Guaraní
    "uk": "ukr",      # Ukrainian
    "bn": "ben",      # Bengali
    "th": "tha",      # Thai
    "he": "heb",      # Hebrew
    "zh-CN": "cmn",   # Mandarin Chinese
    "ar": "arb",      # Arabic Modern Standard
    "it": "ita",      # Italian
    "hi": "hin",      # Hindi
}

def download_file(filename, retries=3):
    """下载单个CSV文件"""
    url = BASE_URL + filename
    output_path = EXTERNAL_WALS_DIR / filename

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

def download_wals_data(force_redownload=False):
    """下载WALS数据集，除非文件已存在且不需要强制重新下载"""
    print("检查WALS数据集...")

    all_exist = all((EXTERNAL_WALS_DIR / f).exists() for f in FILES)

    if all_exist and not force_redownload:
        print("WALS数据集已存在，跳过下载。")
        return True

    print("开始下载WALS数据集...")
    success_count = 0
    for filename in FILES:
        if download_file(filename):
            success_count += 1

    if success_count == len(FILES):
        print(f"✅ WALS数据集下载完成!")
        return True
    else:
        print(f"❌ 下载失败: {success_count}/{len(FILES)} 个文件")
        return False

def load_wals_data():
    """加载WALS数据集到DataFrame"""
    print("加载WALS数据...")

    languages_path = EXTERNAL_WALS_DIR / "languages.csv"
    parameters_path = EXTERNAL_WALS_DIR / "parameters.csv"
    values_path = EXTERNAL_WALS_DIR / "values.csv"
    codes_path = EXTERNAL_WALS_DIR / "codes.csv"

    languages_df = pd.read_csv(languages_path, keep_default_na=False)
    parameters_df = pd.read_csv(parameters_path, keep_default_na=False)
    values_df = pd.read_csv(values_path, keep_default_na=False)
    codes_df = pd.read_csv(codes_path, keep_default_na=False)

    print(f"  语言表: {len(languages_df)} 行")
    print(f"  参数表: {len(parameters_df)} 行")
    print(f"  特征值表: {len(values_df)} 行")
    print(f"  代码表: {len(codes_df)} 行")

    return languages_df, parameters_df, values_df, codes_df

def create_iso_to_wals_map(languages_df):
    """创建ISO 639-3代码到WALS Language_ID的映射"""
    iso_to_wals = {}

    # 首先尝试ISO639P3code列（单个ISO 639-3代码）
    if 'ISO639P3code' in languages_df.columns:
        for _, row in languages_df.iterrows():
            iso_code = row['ISO639P3code']
            if iso_code and pd.notna(iso_code) and str(iso_code).strip():
                code = str(iso_code).strip()
                iso_to_wals[code] = row['ID']
                iso_to_wals[code.lower()] = row['ID']

    # 然后尝试ISO_codes列（可能包含多个代码）
    if 'ISO_codes' in languages_df.columns:
        for _, row in languages_df.iterrows():
            iso_codes_str = row['ISO_codes']
            if iso_codes_str and pd.notna(iso_codes_str) and iso_codes_str.strip():
                # 可能包含多个代码，用空格分隔
                codes = str(iso_codes_str).strip().split()
                for code in codes:
                    iso_to_wals[code] = row['ID']
                    iso_to_wals[code.lower()] = row['ID']

    # 后备方案：soc_ids列
    elif 'soc_ids' in languages_df.columns:
        for _, row in languages_df.iterrows():
            soc_ids_str = row['soc_ids']
            if soc_ids_str and pd.notna(soc_ids_str) and soc_ids_str.strip():
                ids = str(soc_ids_str).strip().split(';')
                for id_str in ids:
                    id_str = id_str.strip()
                    # 检查是否是ISO 639-3代码（3个字母）
                    if len(id_str) == 3 and id_str.isalpha():
                        iso_to_wals[id_str] = row['ID']
                        iso_to_wals[id_str.lower()] = row['ID']

    # 也创建语言名称到WALS ID的映射（用于后备）
    name_to_wals = {}
    for _, row in languages_df.iterrows():
        name = str(row['Name']).strip().lower()
        name_to_wals[name] = row['ID']

    print(f"  创建的ISO代码映射: {len(iso_to_wals)} 个条目")
    return iso_to_wals, name_to_wals

def get_wals_ids_for_languages(target_codes, iso_to_wals_map, name_to_wals_map):
    """获取目标语言的WALS Language_ID"""
    wals_ids = {}
    missing_codes = []

    # 语言名称映射（用于后备）
    language_names = {
        "zu": ["zulu"],
        "gd": ["scottish gaelic", "gaelic"],
        "hmn": ["hmong", "miao"],
        "gn": ["guarani", "paraguayan guarani"],
        "uk": ["ukrainian"],
        "bn": ["bengali", "bangla"],
        "th": ["thai"],
        "he": ["hebrew"],
        "zh-CN": ["chinese", "mandarin", "mandarin chinese", "standard chinese"],
        "ar": ["arabic", "standard arabic"],
        "it": ["italian"],
        "hi": ["hindi"],
    }

    for user_code, iso_code in target_codes.items():
        wals_id = None

        # 尝试ISO 639-3代码
        wals_id = iso_to_wals_map.get(iso_code)
        if wals_id is None:
            wals_id = iso_to_wals_map.get(iso_code.lower())

        if wals_id is None:
            # 尝试用户提供的代码
            wals_id = iso_to_wals_map.get(user_code)
            if wals_id is None:
                wals_id = iso_to_wals_map.get(user_code.lower())

        if wals_id is None:
            # 尝试语言名称映射
            names = language_names.get(user_code, [])
            for name in names:
                wals_id = name_to_wals_map.get(name.lower())
                if wals_id:
                    break

        if wals_id:
            wals_ids[user_code] = wals_id
            print(f"    {user_code} -> {iso_code} -> WALS ID: {wals_id}")
        else:
            missing_codes.append((user_code, iso_code))
            print(f"    ⚠️ {user_code} -> {iso_code}: 未找到WALS ID")

    return wals_ids, missing_codes

def extract_features_for_language(language_code, wals_id, values_df, parameters_df):
    """提取单个语言的所有特征"""
    # 过滤该语言的特征值
    lang_values = values_df[values_df["Language_ID"] == wals_id].copy()

    if lang_values.empty:
        print(f"    {language_code}: 没有找到特征值")
        return pd.DataFrame()

    # 合并特征参数信息获取特征名称
    lang_features = pd.merge(
        lang_values,
        parameters_df[["ID", "Name"]],
        left_on="Parameter_ID",
        right_on="ID",
        how="left",
        suffixes=("", "_param")
    )

    # 重命名列以符合输出格式
    result_df = lang_features[["Parameter_ID", "Value", "Name"]].copy()
    result_df.columns = ["Fid", "Value", "Feature"]

    # 按特征ID排序（保持WALS章节顺序）
    def extract_sort_key(fid):
        # 将"1A"转换为(1, 'A')用于排序
        if isinstance(fid, str):
            # 提取数字部分和字母部分
            match = re.match(r'(\d+)([A-Za-z]*)', fid)
            if match:
                num = int(match.group(1))
                letter = match.group(2) if match.group(2) else ''
                return (num, letter)
        return (9999, '')

    result_df['sort_key'] = result_df['Fid'].apply(extract_sort_key)
    result_df = result_df.sort_values('sort_key').drop('sort_key', axis=1)

    print(f"    {language_code}: 找到 {len(result_df)} 个特征")
    return result_df

def save_language_features(language_code, features_df):
    """保存语言特征到CSV文件"""
    if features_df.empty:
        print(f"    {language_code}: 无特征数据，跳过保存")
        return False

    output_path = WALS_OUTPUT_DIR / f"{language_code}_wals.csv"
    features_df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"    {language_code}: 保存到 {output_path}")
    return True

def main():
    import argparse

    parser = argparse.ArgumentParser(description='从WALS数据库获取语言特征数据')
    parser.add_argument('--force-download', action='store_true',
                       help='强制重新下载WALS数据集')
    parser.add_argument('--skip-download', action='store_true',
                       help='跳过下载，使用现有数据')
    parser.add_argument('--languages', nargs='+',
                       help='指定要处理的语言代码（默认：所有12种语言）')

    args = parser.parse_args()

    print("=" * 60)
    print("WALS语言特征提取工具")
    print("=" * 60)

    # 确定要处理的语言
    if args.languages:
        target_codes = {code: LANGUAGE_MAPPING.get(code, code)
                       for code in args.languages if code in LANGUAGE_MAPPING}
        if len(target_codes) < len(args.languages):
            print(f"警告：部分语言代码未在映射中找到: {set(args.languages) - set(LANGUAGE_MAPPING.keys())}")
    else:
        target_codes = LANGUAGE_MAPPING

    print(f"目标语言: {list(target_codes.keys())}")
    print()

    # 下载数据（如果需要）
    if not args.skip_download:
        if not download_wals_data(args.force_download):
            print("❌ 下载失败，退出")
            sys.exit(1)
    else:
        print("跳过下载，使用现有数据...")

    # 加载数据
    languages_df, parameters_df, values_df, codes_df = load_wals_data()

    # 创建ISO代码到WALS ID的映射
    iso_to_wals_map, name_to_wals_map = create_iso_to_wals_map(languages_df)

    # 获取目标语言的WALS ID
    wals_ids, missing_codes = get_wals_ids_for_languages(
        target_codes, iso_to_wals_map, name_to_wals_map
    )

    if missing_codes:
        print(f"\n⚠️ 警告: {len(missing_codes)} 个语言代码未找到:")
        for user_code, iso_code in missing_codes:
            print(f"  - {user_code} (ISO 639-3: {iso_code})")

        # 显示可能的匹配建议
        print("\n语言表中包含以下相关语言:")
        search_terms = ["zulu", "gaelic", "hmong", "guarani", "ukrainian",
                       "bengali", "thai", "hebrew", "chinese", "arabic",
                       "italian", "hindi"]
        for term in search_terms:
            matches = languages_df[
                languages_df['Name'].astype(str).str.lower().str.contains(term)
            ]
            if not matches.empty:
                for _, row in matches.head(2).iterrows():
                    print(f"  - {row['Name']}: id={row['ID']}, iso_codes={row.get('ISO_codes', 'N/A')}")

    # 为每个找到的语言提取特征
    print(f"\n提取特征...")
    extracted_count = 0

    for language_code, wals_id in wals_ids.items():
        print(f"\n处理 {language_code} (WALS ID: {wals_id})...")

        # 提取特征
        features_df = extract_features_for_language(
            language_code, wals_id, values_df, parameters_df
        )

        # 保存到CSV
        if save_language_features(language_code, features_df):
            extracted_count += 1

    print(f"\n" + "=" * 60)
    print("✅ 完成!")
    print(f"  - 成功提取: {extracted_count}/{len(target_codes)} 种语言")
    print(f"  - 输出目录: {WALS_OUTPUT_DIR.absolute()}")

    if missing_codes:
        print(f"  - 未找到的语言: {len(missing_codes)} 种")

    print("=" * 60)

if __name__ == "__main__":
    main()