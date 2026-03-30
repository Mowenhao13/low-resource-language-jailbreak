#!/usr/bin/env python3
"""
使用GramBank数据库获取语言形态特征
完全依赖GramBank数据，无手动标注
"""

import os
import sys
import pandas as pd

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# 语言ID映射 - 基于GramBank实际存在的ID
LANG_GRAMBANK_MAP = {
    'zu': 'zulu1248',     # Zulu
    'gd': None,            # Scottish Gaelic - not in Grambank
    'hmn': None,           # Hmong - not in Grambank
    'gn': None,            # Guarani - not in Grambank
    'uk': 'ukra1253',     # Ukrainian (corrected)
    'bn': None,            # Bengali - not in Grambank
    'th': 'thai1261',      # Thai (corrected)
    'he': 'hebr1245',     # Modern Hebrew
    'zh-CN': 'mand1415',  # Mandarin Chinese
    'ar': 'stan1318',     # Standard Arabic
    'it': 'ital1282',     # Italian
    'hi': 'hind1269'      # Hindi
}

# 形态相关特征筛选
MORPHOLOGY_FEATURES = [
    # 名词形态
    'GB039',  # Nonphonological allomorphy of noun number markers
    'GB041',  # Suppletive nouns for number
    'GB042',  # Productive morphological singular marking
    'GB043',  # Productive morphological dual marking
    'GB044',  # Productive morphological plural marking
    'GB046',  # Associative plural marker
    'GB051',  # Gender/noun class system (sex)
    'GB052',  # Gender/noun class system (shape)
    'GB053',  # Gender/noun class system (animacy)
    'GB054',  # Gender/noun class system (plant)

    # 格标记
    'GB070',  # Morphological cases for non-pronominal core args
    'GB071',  # Morphological cases for pronominal core args
    'GB072',  # Morphological cases for oblique non-pronominal NPs
    'GB073',  # Morphological cases for independent oblique personal pronouns

    # 动词形态
    'GB079',  # Verb prefixes/proclitics (other than A/S/P)
    'GB080',  # Verb suffixes/enclitics (other than A/S/P)
    'GB081',  # Productive infixation in verbs
    'GB082',  # Overt morphological marking of present tense
    'GB083',  # Overt morphological marking of past tense
    'GB084',  # Overt morphological marking of future tense
    'GB086',  # Perfective/imperfective aspect distinction
    'GB089',  # S argument indexed by suffix/enclitic
    'GB090',  # S argument indexed by prefix/proclitic
    'GB091',  # A argument indexed by suffix/enclitic
    'GB092',  # A argument indexed by prefix/proclitic
    'GB093',  # P argument indexed by suffix/enclitic
    'GB094',  # P argument indexed by prefix/proclitic
    'GB099',  # Verb stem alter according to person
    'GB103',  # Benefactive applicative marker
    'GB104',  # Instrumental applicative marker
    'GB107',  # Standard negation marked by affix/clitic/modification
    'GB108',  # Directional/locative morphological marking on verbs
    'GB109',  # Verb suppletion for participant number
    'GB110',  # Verb suppletion for tense/aspect
    'GB111',  # Conjugation classes
    'GB113',  # Verbal affixes/clitics turning intransitive to transitive
    'GB114',  # Phonologically bound reflexive marker
    'GB115',  # Phonologically bound reciprocal marker
    'GB147',  # Morphological passive
    'GB148',  # Morphological antipassive
    'GB149',  # Morphologically marked inverse
    'GB155',  # Causatives formed by affixes/clitics
]


def load_grambank_data():
    """加载GramBank数据"""
    data_path = os.path.join(project_root, 'data', 'external')

    print("=" * 60)
    print("加载GramBank数据...")
    print("=" * 60)

    # 加载语言数据
    langs_df = pd.read_csv(os.path.join(data_path, 'languages.csv'))
    print(f"  语言数据: {len(langs_df)} 种语言")

    # 加载特征定义
    params_df = pd.read_csv(os.path.join(data_path, 'parameters.csv'))
    print(f"  特征定义: {len(params_df)} 个特征")

    # 加载特征值 - 完整加载
    print("  正在加载特征值...")
    values_df = pd.read_csv(os.path.join(data_path, 'values.csv'))
    print(f"  特征值记录: {len(values_df)} 条")

    # 加载代码映射
    codes_df = pd.read_csv(os.path.join(data_path, 'codes.csv'))
    print(f"  代码映射: {len(codes_df)} 条记录")

    return langs_df, params_df, values_df, codes_df


def calculate_morphological_complexity(features):
    """
    基于GramBank特征计算形态复杂度和类型
    返回: (morph_type, complexity_score, confidence)
    """
    has_case = False
    has_verb_affix = False
    has_gender = False
    has_number_marking = False

    # 分析格标记
    case_features = ['GB070', 'GB071', 'GB072', 'GB073']
    for feat in case_features:
        if feat in features and features[feat]['value'] == '1':
            has_case = True
            break

    # 分析动词形态
    verb_features = ['GB079', 'GB080', 'GB082', 'GB083', 'GB084', 'GB086',
                     'GB089', 'GB090', 'GB091', 'GB092', 'GB093', 'GB094']
    verb_count = 0
    for feat in verb_features:
        if feat in features and features[feat]['value'] == '1':
            verb_count += 1
    if verb_count >= 3:
        has_verb_affix = True

    # 分析性别/名词类
    gender_features = ['GB051', 'GB052', 'GB053', 'GB054']
    for feat in gender_features:
        if feat in features and features[feat]['value'] == '1':
            has_gender = True
            break

    # 分析数标记
    number_features = ['GB042', 'GB043', 'GB044']
    for feat in number_features:
        if feat in features and features[feat]['value'] == '1':
            has_number_marking = True
            break

    # 计算形态标记总数（用于复杂度评分）
    total_markers = 0
    for feat_id, feat_data in features.items():
        if feat_data['value'] == '1':
            total_markers += 1

    # 确定形态类型
    if has_case and has_verb_affix and has_gender:
        morph_type = 'fusional'
        confidence = 0.9
    elif has_verb_affix and has_number_marking and not has_case:
        morph_type = 'agglutinative'
        confidence = 0.8
    elif not has_case and not has_verb_affix and not has_gender:
        morph_type = 'isolating'
        confidence = 0.9
    elif total_markers >= 10:
        morph_type = 'fusional'
        confidence = 0.7
    elif 3 <= total_markers < 10:
        morph_type = 'agglutinative'
        confidence = 0.6
    else:
        morph_type = 'isolating'
        confidence = 0.5

    # 复杂度评分 (0-2)
    complexity_score = min(total_markers / 20, 1.0) + (0.5 if has_case else 0) + (0.5 if has_gender else 0)
    complexity_score = min(complexity_score, 2.0)

    return morph_type, complexity_score, confidence


def extract_features_for_language(lang_id, values_df, codes_df):
    """提取指定语言的所有形态相关特征"""
    # 获取该语言的所有特征值
    lang_values = values_df[values_df['Language_ID'] == lang_id].copy()

    if len(lang_values) == 0:
        return {}

    # 筛选形态相关特征
    morph_values = lang_values[lang_values['Parameter_ID'].isin(MORPHOLOGY_FEATURES)]

    features = {}

    for _, row in morph_values.iterrows():
        param_id = row['Parameter_ID']
        value = row['Value']
        code_id = row.get('Code_ID')

        # 跳过未知值
        if pd.isna(value) or value == '?':
            continue

        # 获取特征值的含义
        description = str(value)
        if pd.notna(code_id) and code_id:
            code_info = codes_df[codes_df['ID'] == code_id]
            if len(code_info) > 0:
                description = code_info.iloc[0].get('Description', str(value))

        features[param_id] = {
            'value': value,
            'description': description
        }

    return features


def main():
    """主函数"""
    langs_df, params_df, values_df, codes_df = load_grambank_data()

    print("\n" + "=" * 60)
    print("提取语言形态特征（基于GramBank）")
    print("=" * 60)

    results = []

    for lang_code, grambank_id in LANG_GRAMBANK_MAP.items():
        print(f"\n处理语言: {lang_code}")

        # 检查GramBank ID是否存在
        if grambank_id is None:
            print(f"  ⚠️  不在GramBank覆盖范围内")
            results.append({
                'language_code': lang_code,
                'grambank_id': None,
                'found_in_grambank': False,
                'morph_type': 'unknown',
                'morph_confidence': 0.0,
                'source': 'GramBank'
            })
            continue

        # 查找语言在GramBank中的信息
        lang_info = langs_df[langs_df['ID'] == grambank_id]

        if len(lang_info) == 0:
            print(f"  ⚠️  未在GramBank中找到: {grambank_id}")
            results.append({
                'language_code': lang_code,
                'grambank_id': grambank_id,
                'found_in_grambank': False,
                'morph_type': 'unknown',
                'morph_confidence': 0.0,
                'source': 'GramBank'
            })
            continue

        lang_info = lang_info.iloc[0]
        lang_name = lang_info['Name']
        print(f"  ✓ 找到: {lang_name}")

        # 提取形态特征
        features = extract_features_for_language(grambank_id, values_df, codes_df)
        print(f"  形态特征数量: {len(features)}")

        # 推断形态类型和复杂度
        morph_type, complexity_score, confidence = calculate_morphological_complexity(features)
        print(f"  推断形态类型: {morph_type} (置信度: {confidence:.2f})")
        print(f"  复杂度评分: {complexity_score:.2f}")

        # 显示一些关键特征
        key_features = ['GB070', 'GB079', 'GB083', 'GB084', 'GB051', 'GB044']
        print("  关键特征:")
        for feat_id in key_features:
            if feat_id in features:
                feat_name = params_df[params_df['ID'] == feat_id]['Name'].iloc[0] if len(params_df[params_df['ID'] == feat_id]) > 0 else feat_id
                feat_name = feat_name[:40] + '...' if len(feat_name) > 40 else feat_name
                print(f"    {feat_id}: {features[feat_id]['description']}")

        results.append({
            'language_code': lang_code,
            'grambank_id': grambank_id,
            'grambank_name': lang_name,
            'found_in_grambank': True,
            'morph_type': morph_type,
            'morph_complexity_score': complexity_score,
            'morph_confidence': confidence,
            'morph_feature_count': len(features),
            'source': 'GramBank'
        })

    # 保存结果
    df = pd.DataFrame(results)
    output_path = os.path.join(project_root, 'data', 'processed', 'morphology_features.csv')
    df.to_csv(output_path, index=False, encoding='utf-8')

    print("\n" + "=" * 60)
    print(f"结果已保存到: {output_path}")
    print("=" * 60)

    # 统计摘要
    print("\n统计摘要:")
    print(f"  总语言数: {len(results)}")
    print(f"  在GramBank中找到: {sum(1 for r in results if r['found_in_grambank'])}")

    print("\n形态类型分布:")
    type_counts = df['morph_type'].value_counts()
    for morph_type, count in type_counts.items():
        print(f"  {morph_type:15s}: {count}")

    print(f"\n平均置信度: {df['morph_confidence'].mean():.3f}")

    return df


if __name__ == "__main__":
    main()