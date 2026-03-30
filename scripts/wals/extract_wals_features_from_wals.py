#!/usr/bin/env python3
"""
从WALS数据集中提取指定语言的特征值（使用真正的WALS数据集）。

输入：12种语言的ISO 639-1/3代码
输出：每个语言的CSV文件，包含Fid, Value, Feature三列

数据文件位于：data/external_wals/
输出目录：data/wals/
"""

import pandas as pd
import os
from pathlib import Path

# 配置路径
DATA_DIR = Path("data/external_wals")
OUTPUT_DIR = Path("data/wals")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 语言代码映射：用户提供的代码 -> ISO 639-3代码
# 注意：有些是ISO 639-1，有些是ISO 639-3
# 基于WALS数据集的实际ISO代码
LANGUAGE_MAPPING = {
    "zu": "zul",      # Zulu
    "gd": "gla",      # Scottish Gaelic (WALS ID: gae)
    "hmn": "hnj",     # Hmong Njua (ISO: hnj) - 替代 hmong
    "gn": "gug",      # Guaraní (ISO: gug) - 替代 grn
    "uk": "ukr",      # Ukrainian
    "bn": "ben",      # Bengali
    "th": "tha",      # Thai
    "he": "heb",      # Hebrew
    "zh-CN": "cmn",   # Mandarin Chinese (ISO: cmn) - 替代 zho
    "ar": "arb",      # Arabic Modern Standard (ISO: arb) - 替代 ara
    "it": "ita",      # Italian
    "hi": "hin",      # Hindi
}

def load_wals_data():
    """加载WALS数据集"""
    print("加载WALS数据...")

    # 读取语言表
    languages_path = DATA_DIR / "languages.csv"
    languages_df = pd.read_csv(languages_path, keep_default_na=False)
    print(f"  语言表: {len(languages_df)} 行, 列: {list(languages_df.columns)}")

    # 读取特征参数表
    parameters_path = DATA_DIR / "parameters.csv"
    parameters_df = pd.read_csv(parameters_path, keep_default_na=False)
    print(f"  参数表: {len(parameters_df)} 行")

    # 读取特征值表
    values_path = DATA_DIR / "values.csv"
    values_df = pd.read_csv(values_path, keep_default_na=False)
    print(f"  特征值表: {len(values_df)} 行")

    # 读取代码表（可选，用于值标签）
    codes_path = DATA_DIR / "codes.csv"
    codes_df = pd.read_csv(codes_path, keep_default_na=False)
    print(f"  代码表: {len(codes_df)} 行")

    return languages_df, parameters_df, values_df, codes_df

def create_language_id_map(languages_df):
    """创建ISO 639-3代码到WALS Language_ID的映射"""
    # WALS languages.csv可能有'iso_codes'或'soc_ids'列
    # 查看列名
    print("  语言表列名:", list(languages_df.columns))

    # 尝试查找包含ISO代码的列
    iso_cols = [col for col in languages_df.columns if 'iso' in col.lower() or 'code' in col.lower()]
    print(f"  可能的ISO代码列: {iso_cols}")

    # WALS可能使用'soc_ids'列包含ISO代码
    # 先尝试'iso_codes'列
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

    elif 'soc_ids' in languages_df.columns:
        # soc_ids可能包含ISO代码
        for _, row in languages_df.iterrows():
            soc_ids_str = row['soc_ids']
            if soc_ids_str and pd.notna(soc_ids_str) and soc_ids_str.strip():
                # 可能包含多种ID，用分号分隔
                ids = str(soc_ids_str).strip().split(';')
                for id_str in ids:
                    id_str = id_str.strip()
                    # 检查是否是ISO 639-3代码（3个字母）
                    if len(id_str) == 3 and id_str.isalpha():
                        iso_to_wals[id_str] = row['ID']
                        iso_to_wals[id_str.lower()] = row['ID']

    # 也尝试通过语言名称映射（后备方案）
    name_to_wals = {}
    for _, row in languages_df.iterrows():
        name = str(row['Name']).strip().lower()
        name_to_wals[name] = row['ID']

    print(f"  创建的ISO代码映射: {len(iso_to_wals)} 个条目")
    print(f"  创建的语言名称映射: {len(name_to_wals)} 个条目")

    return iso_to_wals, name_to_wals

def get_language_wals_ids(target_codes, iso_to_wals_map, name_to_wals_map):
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
    # 先按数字部分排序
    def extract_sort_key(fid):
        # 将"1A"转换为(1, 'A')用于排序
        if isinstance(fid, str):
            # 提取数字部分和字母部分
            import re
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

    output_path = OUTPUT_DIR / f"{language_code}_wals.csv"
    features_df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"    {language_code}: 保存到 {output_path}")
    return True

def main():
    print("开始提取WALS语言特征...")
    print(f"数据目录: {DATA_DIR.absolute()}")
    print(f"输出目录: {OUTPUT_DIR.absolute()}\n")

    # 加载数据
    languages_df, parameters_df, values_df, codes_df = load_wals_data()

    # 创建ISO代码到WALS ID的映射
    iso_to_wals_map, name_to_wals_map = create_language_id_map(languages_df)

    # 获取目标语言的WALS ID
    target_codes = LANGUAGE_MAPPING
    wals_ids, missing_codes = get_language_wals_ids(
        target_codes, iso_to_wals_map, name_to_wals_map
    )

    if missing_codes:
        print(f"\n⚠️ 警告: {len(missing_codes)} 个语言代码未找到:")
        for user_code, iso_code in missing_codes:
            print(f"  - {user_code} (ISO 639-3: {iso_code})")
        print("尝试在languages.csv中搜索替代代码...")

        # 显示一些可能的匹配
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

    print(f"\n完成!")
    print(f"  - 成功提取: {extracted_count}/{len(target_codes)} 种语言")
    print(f"  - 输出目录: {OUTPUT_DIR.absolute()}")

    if missing_codes:
        print(f"  - 未找到的语言: {len(missing_codes)} 种")
        print("\n建议:")
        print("  1. 检查languages.csv中的iso_codes或soc_ids列")
        print("  2. 在WALS网站查找正确的语言代码")
        print("  3. 更新LANGUAGE_MAPPING字典或添加备用名称")

if __name__ == "__main__":
    main()