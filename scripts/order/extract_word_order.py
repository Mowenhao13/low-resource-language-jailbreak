#!/usr/bin/env python3
"""
从WALS数据中提取语序特征
包括：基本语序(SVO/SOV等)、附置词位置、形容词-名词顺序等
"""

import pandas as pd
import os
from pathlib import Path

# 配置路径
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
WALS_DIR = DATA_DIR / "wals"
OUTPUT_DIR = DATA_DIR / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 目标语言
LANGUAGES = ["zu", "gd", "hmn", "gn", "uk", "bn", "th", "he", "zh-CN", "ar", "it", "hi"]

# 语序特征定义
WORD_ORDER_FEATURES = {
    "81A": {
        "name": "basic_word_order",
        "description": "主语、宾语、动词的基本顺序",
        "value_map": {
            1: "SOV",
            2: "SVO",
            3: "VSO",
            4: "VOS",
            5: "OVS",
            6: "OSV",
            7: "no_dominant_order"
        }
    },
    "81B": {
        "name": "two_dominant_orders",
        "description": "有两种主导语序",
        "value_map": {
            1: "no",
            2: "yes"
        }
    },
    "82A": {
        "name": "subject_verb_order",
        "description": "主语和动词的顺序",
        "value_map": {
            1: "SV",
            2: "VS"
        }
    },
    "83A": {
        "name": "object_verb_order",
        "description": "宾语和动词的顺序",
        "value_map": {
            1: "OV",
            2: "VO"
        }
    },
    "84A": {
        "name": "object_oblique_verb_order",
        "description": "宾语、旁格和动词的顺序"
    },
    "85A": {
        "name": "adposition_order",
        "description": "附置词与名词短语的顺序",
        "value_map": {
            1: "prepositions",
            2: "postpositions",
            3: "inpositions",
            4: "no_adpositions"
        }
    },
    "86A": {
        "name": "genitive_noun_order",
        "description": "领属词与名词的顺序",
        "value_map": {
            1: "genitive-noun",
            2: "noun-genitive"
        }
    },
    "87A": {
        "name": "adjective_noun_order",
        "description": "形容词与名词的顺序",
        "value_map": {
            1: "adjective-noun",
            2: "noun-adjective",
            3: "both_orders_no_dominant",
            4: "no_adjectives_distinct_class"
        }
    },
    "88A": {
        "name": "demonstrative_noun_order",
        "description": "指示词与名词的顺序",
        "value_map": {
            1: "demonstrative-noun",
            2: "noun-demonstrative",
            3: "both_orders"
        }
    },
    "89A": {
        "name": "numeral_noun_order",
        "description": "数词与名词的顺序",
        "value_map": {
            1: "numeral-noun",
            2: "noun-numeral",
            3: "both_orders"
        }
    },
    "90A": {
        "name": "relative_clause_noun_order",
        "description": "关系从句与名词的顺序",
        "value_map": {
            1: "noun-relativizer",
            2: "relativizer-noun",
            3: "correlative",
            4: "internally_headed",
            5: "adjoined"
        }
    }
}

def load_wals_codes():
    """加载WALS代码映射表"""
    codes_path = DATA_DIR / "external_wals" / "codes.csv"
    codes_df = pd.read_csv(codes_path, keep_default_na=False)
    return codes_df

def extract_feature_for_language(lang_code, feature_id):
    """从语言CSV中提取特定特征"""
    csv_path = WALS_DIR / f"{lang_code}_wals.csv"

    if not csv_path.exists():
        return None

    df = pd.read_csv(csv_path)
    row = df[df['Fid'] == feature_id]

    if row.empty:
        return None

    value = int(row.iloc[0]['Value'])
    feature_name = row.iloc[0]['Feature']

    return {
        'value': value,
        'feature_name': feature_name
    }

def get_value_description(feature_id, value, codes_df):
    """获取特征值的文字描述"""
    if feature_id not in WORD_ORDER_FEATURES:
        return str(value)

    # 首先尝试预定义的映射
    feat_config = WORD_ORDER_FEATURES[feature_id]
    if 'value_map' in feat_config and value in feat_config['value_map']:
        return feat_config['value_map'][value]

    # 然后尝试从codes.csv获取
    code_row = codes_df[
        (codes_df['Parameter_ID'] == feature_id) &
        (codes_df['Number'] == value)
    ]

    if not code_row.empty:
        return code_row.iloc[0]['Name']

    return str(value)

def extract_all_word_order_features():
    """提取所有语言的语序特征"""
    print("提取语序特征...")
    print(f"目标语言: {LANGUAGES}")
    print()

    # 加载代码映射
    codes_df = load_wals_codes()

    results = []

    for lang_code in LANGUAGES:
        print(f"处理语言: {lang_code}")

        lang_data = {'language_code': lang_code}

        for feature_id, feat_config in WORD_ORDER_FEATURES.items():
            feature_data = extract_feature_for_language(lang_code, feature_id)

            if feature_data:
                value = feature_data['value']
                description = get_value_description(feature_id, value, codes_df)

                # 保存原始值和描述
                lang_data[f"{feat_config['name']}_value"] = value
                lang_data[f"{feat_config['name']}_description"] = description
                lang_data[f"{feat_config['name']}_available"] = True

                # 特殊处理：基本语序
                if feature_id == "81A":
                    lang_data['basic_word_order'] = description
            else:
                lang_data[f"{feat_config['name']}_value"] = None
                lang_data[f"{feat_config['name']}_description"] = None
                lang_data[f"{feat_config['name']}_available"] = False

        results.append(lang_data)
        print(f"  完成")

    # 转换为DataFrame
    df = pd.DataFrame(results)

    # 重新排列列，使重要信息在前
    cols = ['language_code', 'basic_word_order', 'basic_word_order_value',
            'subject_verb_order_description', 'object_verb_order_description',
            'adposition_order_description', 'adjective_noun_order_description']

    # 添加其他列
    other_cols = [c for c in df.columns if c not in cols]
    df = df[cols + other_cols]

    return df

def analyze_word_order_patterns(df):
    """分析语序模式"""
    print("\n" + "="*60)
    print("语序特征分析")
    print("="*60)

    # 基本语序分布
    basic_order_counts = df['basic_word_order'].value_counts()
    print(f"\n基本语序分布 (共{len(df)}种语言):")
    for order, count in basic_order_counts.items():
        print(f"  {order}: {count}")

    # 计算语序和谐性
    print("\n语序和谐性分析:")

    # SVO语言的前置词比例
    svo_langs = df[df['basic_word_order'] == 'SVO']
    if len(svo_langs) > 0:
        prepos_count = sum(1 for d in svo_langs['adposition_order_description']
                          if d == 'prepositions')
        print(f"  SVO语言 ({len(svo_langs)}种): {prepos_count/len(svo_langs)*100:.1f}% 使用前置词")

    # SOV语言的后置词比例
    sov_langs = df[df['basic_word_order'] == 'SOV']
    if len(sov_langs) > 0:
        postpos_count = sum(1 for d in sov_langs['adposition_order_description']
                           if d == 'postpositions')
        print(f"  SOV语言 ({len(sov_langs)}种): {postpos_count/len(sov_langs)*100:.1f}% 使用后置词")

    # 形容词-名词顺序分布
    adj_noun_counts = df['adjective_noun_order_description'].value_counts()
    print(f"\n形容词-名词顺序分布:")
    for order, count in adj_noun_counts.items():
        print(f"  {order}: {count}")

def main():
    """主函数"""
    print("="*60)
    print("WALS语序特征提取工具")
    print("="*60)

    # 提取特征
    df = extract_all_word_order_features()

    # 分析模式
    analyze_word_order_patterns(df)

    # 保存结果
    output_path = OUTPUT_DIR / "word_order_features.csv"
    df.to_csv(output_path, index=False, encoding='utf-8')

    print(f"\n" + "="*60)
    print(f"✅ 完成!")
    print(f"结果保存到: {output_path}")
    print(f"共处理 {len(df)} 种语言")
    print(f"提取 {len(WORD_ORDER_FEATURES)} 个语序特征")

    # 显示摘要
    print("\n语言基本语序摘要:")
    for _, row in df.iterrows():
        lang_code = row['language_code']
        basic_order = row.get('basic_word_order', 'unknown')
        adj_order = row.get('adjective_noun_order_description', 'unknown')
        print(f"  {lang_code}: {basic_order}, 形名顺序: {adj_order}")

    print("="*60)

if __name__ == "__main__":
    main()