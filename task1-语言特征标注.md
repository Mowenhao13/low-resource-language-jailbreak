# Task 1: 语言特征标注详细执行方案

## 项目背景
基于论文"Low-Resource Languages Jailbreak GPT-4"的研究，本项目需要为12种语言标注6个关键语言特征，以分析语言特征与越狱成功率(ASR)的相关性。本文件详细描述每个特征的获取与标注执行步骤。

## 语言列表
基于论文中的12种语言，分为低资源(LRL)、中资源(MRL)、高资源(HRL)三类：

### 低资源语言 (LRL)
1. **祖鲁语 (Zulu)** - 语言代码: `zu`
2. **苏格兰盖尔语 (Scots Gaelic)** - 语言代码: `gd`
3. **苗语 (Hmong)** - 语言代码: `hmn`
4. **瓜拉尼语 (Guarani)** - 语言代码: `gn`

### 中资源语言 (MRL)
5. **乌克兰语 (Ukrainian)** - 语言代码: `uk`
6. **孟加拉语 (Bengali)** - 语言代码: `bn`
7. **泰语 (Thai)** - 语言代码: `th`
8. **希伯来语 (Hebrew)** - 语言代码: `he`

### 高资源语言 (HRL)
9. **简体中文 (Simplified Mandarin Chinese)** - 语言代码: `zh-CN`
10. **现代标准阿拉伯语 (Modern Standard Arabic)** - 语言代码: `ar`
11. **意大利语 (Italian)** - 语言代码: `it`
12. **印地语 (Hindi)** - 语言代码: `hi`

---

## 1. 形态复杂度标注

### 定义
标注语言的形态类型：孤立语、屈折语、黏着语或混合类型。

### 数据来源
1. **WALS数据库** (World Atlas of Language Structures) - 特征26A: Fusion of Selected Inflectional Formatives
2. **Ethnologue** - 语言分类信息
3. **Glottolog** - 语言谱系和类型学数据
4. **语言学文献** - 学术论文和专著

### 执行步骤

#### 步骤1: 环境准备和数据收集
```bash
# 安装WALS Python库（如果可用）
pip install pywals wals-data

# 下载WALS形态特征数据集
wget https://wals.info/feature/26A.csv -O data/raw/wals_morphology.csv

# 或直接使用API查询
```

#### 步骤2: 自动化查询脚本
创建 `scripts/extract_morphology.py`:

```python
import pandas as pd
import requests

def get_morphological_type(lang_code):
    """通过WALS API获取形态类型"""
    # WALS API端点（示例）
    url = f"https://wals.info/feature/26A.json?language={lang_code}"
    response = requests.get(url)
    data = response.json()

    # 解析结果
    wals_code = data.get('value', '')
    mapping = {
        '1': ('isolating', 0.9),      # 孤立语
        '2': ('fusional', 0.9),       # 屈折语
        '3': ('agglutinative', 0.9),  # 黏着语
        '4': ('mixed', 0.7),          # 混合型
    }

    morph_type, confidence = mapping.get(wals_code, ('unknown', 0.5))

    return {
        'morph_type': morph_type,
        'morph_confidence': confidence,
        'wals_code': wals_code,
        'source': 'WALS'
    }

# 批量处理所有语言
languages = ['zu', 'gd', 'hmn', 'gn', 'uk', 'bn', 'th', 'he', 'zh-CN', 'ar', 'it', 'hi']
for lang in languages:
    result = get_morphological_type(lang)
    print(f"{lang}: {result}")
```

#### 步骤3: 备用方案（手动标注）
创建 `config/manual_morphology.yaml` 用于手动标注缺失数据:

```yaml
manual_annotations:
  zu:
    morph_type: "agglutinative"
    confidence: 0.85
    notes: "班图语系典型黏着语"
  gd:
    morph_type: "fusional"
    confidence: 0.8
    notes: "凯尔特语族，有一定屈折特征"
  hmn:
    morph_type: "isolating"
    confidence: 0.75
    notes: "苗瑶语系，偏孤立语特征"
  gn:
    morph_type: "agglutinative"
    confidence: 0.7
    notes: "图皮-瓜拉尼语系，黏着特征"
  uk:
    morph_type: "fusional"
    confidence: 0.9
    notes: "斯拉夫语族，典型屈折语"
  # ... 其他语言
```

### 特征变量命名
- `morph_type`: 形态类型 (isolating/fusional/agglutinative/mixed/unknown)
- `morph_confidence`: 置信度 (0-1)
- `morph_source`: 数据来源 (WALS/Ethnologue/Glottolog/Manual)

### 验证命令
```bash
# 测试形态提取脚本
python scripts/extract_morphology.py

# 查看结果
head -20 data/processed/morphology_features.csv
```

---

## 2. 词序灵活度标注

### 定义
基本词序(SVO/SOV/VSO等)和词序变化灵活度。

### 数据来源
1. **WALS特征81A-87A** - 词序相关特征
2. **Universal Dependencies (UD) 树库** - 实际语料分析
3. **语言学文献** - 特定语言研究

### 执行步骤

#### 步骤1: 基础词序获取
```bash
# 下载WALS词序数据
wget https://wals.info/feature/82A.csv -O data/raw/wals_word_order.csv

# 下载UD树库（示例，根据语言可用性）
wget https://github.com/UniversalDependencies/UD_English-EWT/archive/refs/heads/main.zip
# 其他语言类似
```

#### 步骤2: 灵活度分析脚本
创建 `scripts/analyze_word_order.py`:

```python
import numpy as np
from collections import Counter

def analyze_word_order_flexibility(ud_treebank_path):
    """分析UD树库中的词序分布"""
    # 解析树库文件（需要实际实现parse_ud_treebank函数）
    sentences = parse_ud_treebank(ud_treebank_path)

    word_orders = []
    for sent in sentences:
        # 提取主语(S)、动词(V)、宾语(O)位置
        order = extract_svo_order(sent)
        word_orders.append(order)

    # 统计分布
    order_counts = Counter(word_orders)
    total = sum(order_counts.values())

    # 计算灵活度指标
    if total > 0:
        # 熵值（信息论度量灵活度）
        entropy = -sum((count/total) * np.log2(count/total)
                      for count in order_counts.values())

        # 主词序比例
        main_order, main_count = order_counts.most_common(1)[0]
        main_proportion = main_count / total

        return {
            'basic_order': main_order,
            'order_entropy': entropy,
            'order_flexibility': 1 - main_proportion,  # 灵活度=1-主词序比例
            'order_distribution': dict(order_counts)
        }

    return {'basic_order': 'unknown', 'order_entropy': 0, 'order_flexibility': 0}

def get_word_order_from_wals(lang_code):
    """从WALS获取基础词序（备用方案）"""
    import requests

    # 查询WALS特征82A
    url = f"https://wals.info/feature/82A.json?language={lang_code}"
    response = requests.get(url)
    data = response.json()

    wals_code = data.get('value', '')
    order_mapping = {
        '1': 'SOV', '2': 'SVO', '3': 'VSO',
        '4': 'VOS', '5': 'OVS', '6': 'OSV'
    }

    return {
        'basic_order': order_mapping.get(wals_code, 'unknown'),
        'source': 'WALS'
    }
```

#### 步骤3: 批量处理
```python
# scripts/batch_word_order.py
import pandas as pd
from analyze_word_order import get_word_order_from_wals

languages = ['zu', 'gd', 'hmn', 'gn', 'uk', 'bn', 'th', 'he', 'zh-CN', 'ar', 'it', 'hi']

results = []
for lang in languages:
    # 优先使用UD树库，若无则使用WALS
    try:
        ud_path = f"data/raw/ud_{lang}.conllu"
        result = analyze_word_order_flexibility(ud_path)
        result['language'] = lang
        result['source'] = 'UD'
    except FileNotFoundError:
        result = get_word_order_from_wals(lang)
        result['language'] = lang

    results.append(result)

# 保存结果
df = pd.DataFrame(results)
df.to_csv('data/processed/word_order_features.csv', index=False)
```

### 特征变量命名
- `basic_word_order`: 基本词序 (SVO/SOV/VSO/VOS/OVS/OSV/unknown)
- `order_entropy`: 词序熵值 (0-∞，值越大越灵活)
- `order_flexibility`: 灵活度得分 (0-1)
- `order_source`: 数据来源 (UD/WALS/Manual)

### 验证命令
```bash
# 测试词序分析
python scripts/batch_word_order.py

# 查看结果统计
python -c "import pandas as pd; df=pd.read_csv('data/processed/word_order_features.csv'); print(df.describe())"
```

---

## 3. 与英语的语法距离

### 定义
基于语法特征相似度的距离度量，反映语言与英语的语法差异。

### 数据来源
1. **WALS特征矩阵** - 选择20个核心语法特征
2. **ASJP数据库** - Automated Similarity Judgment Program
3. **语法特征文献** - 语言学参考资料

### 执行步骤

#### 步骤1: 特征选择
创建 `config/grammar_features.yaml` 定义核心特征:

```yaml
selected_features:
  - feature: "26A"   # 形态融合度
    description: "Fusion of Selected Inflectional Formatives"
  - feature: "49A"   # 名词类别数
    description: "Number of Genders"
  - feature: "51A"   # 前置词vs后置词
    description: "Position of Case Affixes"
  - feature: "69A"   # 过去时标记
    description: "Past Tense"
  - feature: "70A"   # 将来时标记
    description: "Future Tense"
  - feature: "81A"   # 词序
    description: "Order of Subject, Object and Verb"
  - feature: "85A"   # 疑问词位置
    description: "Order of Adposition and Noun Phrase"
  - feature: "87A"   # 形容词与名词顺序
    description: "Order of Adjective and Noun"
  - feature: "93A"   # 时态标记
    description: "Position of Tense-Aspect Affixes"
  - feature: "102A"  # 复数标记
    description: "Coding of Nominal Plurality"
  # ... 共选择20个特征
```

#### 步骤2: 距离计算脚本
创建 `scripts/calculate_grammar_distance.py`:

```python
import numpy as np
import requests
import yaml

def get_wals_feature(lang_code, feature_code):
    """查询WALS获取特定特征值"""
    url = f"https://wals.info/feature/{feature_code}.json?language={lang_code}"
    try:
        response = requests.get(url, timeout=10)
        data = response.json()
        return data.get('value', '0')  # 0表示缺失或无关
    except:
        return '0'

def create_feature_vector(lang_code, selected_features):
    """为语言创建二进制特征向量"""
    vector = []
    for feature in selected_features:
        value = get_wals_feature(lang_code, feature['feature'])
        # 转换为二进制：有该特征=1，无/缺失=0
        binary = 1 if value and value != '0' else 0
        vector.append(binary)
    return np.array(vector)

def calculate_grammar_distance(lang1_code, lang2_code, selected_features):
    """计算两种语言的语法距离"""
    vec1 = create_feature_vector(lang1_code, selected_features)
    vec2 = create_feature_vector(lang2_code, selected_features)

    # Jaccard距离
    intersection = np.sum(vec1 & vec2)
    union = np.sum(vec1 | vec2)
    jaccard_dist = 1 - (intersection / union) if union > 0 else 1

    # Hamming距离（标准化）
    hamming_dist = np.sum(vec1 != vec2) / len(vec1)

    # 综合距离（平均值）
    composite_dist = (jaccard_dist + hamming_dist) / 2

    return {
        'jaccard_distance': round(jaccard_dist, 3),
        'hamming_distance': round(hamming_dist, 3),
        'composite_distance': round(composite_dist, 3),
        'features_compared': len(selected_features)
    }

# 主执行函数
def main():
    # 加载特征配置
    with open('config/grammar_features.yaml') as f:
        config = yaml.safe_load(f)
    selected_features = config['selected_features']

    # 计算所有语言与英语的距离
    english_code = 'en'
    target_languages = ['zu', 'gd', 'hmn', 'gn', 'uk', 'bn', 'th', 'he', 'zh-CN', 'ar', 'it', 'hi']

    results = []
    for lang in target_languages:
        distance = calculate_grammar_distance(lang, english_code, selected_features)
        distance['language'] = lang
        results.append(distance)

    # 保存结果
    import pandas as pd
    df = pd.DataFrame(results)
    df.to_csv('data/processed/grammar_distance_features.csv', index=False)
    print(f"Saved grammar distance for {len(df)} languages")

if __name__ == "__main__":
    main()
```

#### 步骤3: 备选距离算法
对于某些语言WALS数据缺失的情况，可以使用ASJP的LDND距离作为补充:

```python
def calculate_asjp_distance(lang1_code, lang2_code):
    """使用ASJP数据库计算音韵距离（作为语法距离的代理）"""
    # ASJP距离通常基于核心词汇的相似度
    # 这里简化实现，实际需要访问ASJP数据库
    asjp_distances = {
        ('en', 'zu'): 0.85, ('en', 'gd'): 0.45, ('en', 'hmn'): 0.92,
        ('en', 'gn'): 0.88, ('en', 'uk'): 0.62, ('en', 'bn'): 0.78,
        ('en', 'th'): 0.82, ('en', 'he'): 0.75, ('en', 'zh-CN'): 0.82,
        ('en', 'ar'): 0.77, ('en', 'it'): 0.28, ('en', 'hi'): 0.68
    }

    key = (lang1_code, lang2_code) if (lang1_code, lang2_code) in asjp_distances else (lang2_code, lang1_code)
    return asjp_distances.get(key, 0.7)
```

### 特征变量命名
- `grammar_distance_jaccard`: Jaccard距离 (0-1，0表示完全相同)
- `grammar_distance_hamming`: Hamming距离 (0-1)
- `grammar_distance_composite`: 综合距离 (0-1)
- `grammar_features_count`: 参与比较的特征数量

### 验证命令
```bash
# 测试语法距离计算
python scripts/calculate_grammar_distance.py

# 查看距离分布
python -c "import pandas as pd; df=pd.read_csv('data/processed/grammar_distance_features.csv'); print(df.sort_values('composite_distance'))"
```

---

## 4. 模型训练数据中的预估占比

### 定义
在大型语言模型(LLM)训练数据中的相对比例估计。

### 数据来源
1. **Common Crawl统计** - 网页语言分布
2. **Wikipedia大小** - 各语言维基百科文章数和字节数
3. **OPUS语料库** - 多语言平行语料统计
4. **学术文献** - 相关研究中的比例估计

### 执行步骤

#### 步骤1: 多源数据收集
```bash
# 下载Wikipedia统计（示例日期）
wget https://dumps.wikimedia.org/other/mediacounts/daily/2026/2026-03/pagecounts-20260301-000000.gz -O data/raw/wikipedia_stats.gz

# 解压并处理
gunzip data/raw/wikipedia_stats.gz

# 下载Common Crawl语言统计（如需）
# 注意：Common Crawl数据量很大，建议使用抽样数据或预计算统计
```

#### 步骤2: 比例估算脚本
创建 `scripts/estimate_training_proportion.py`:

```python
import numpy as np
import pandas as pd
import requests
from math import log

def get_wikipedia_stats(lang_code):
    """获取Wikipedia文章统计"""
    # 映射语言代码到Wikipedia项目代码
    wiki_codes = {
        'zu': 'zu', 'gd': 'gd', 'hmn': 'hmn', 'gn': 'gn',
        'uk': 'uk', 'bn': 'bn', 'th': 'th', 'he': 'he',
        'zh-CN': 'zh', 'ar': 'ar', 'it': 'it', 'hi': 'hi'
    }

    wiki_code = wiki_codes.get(lang_code, lang_code)

    # 查询Wikipedia API
    url = f"https://{wiki_code}.wikipedia.org/w/api.php"
    params = {
        'action': 'query',
        'meta': 'siteinfo',
        'siprop': 'statistics',
        'format': 'json'
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        stats = data['query']['statistics']

        return {
            'article_count': int(stats.get('articles', 0)),
            'total_bytes': int(stats.get('totalbytes', 0)),
            'edits': int(stats.get('edits', 0))
        }
    except:
        return {'article_count': 0, 'total_bytes': 0, 'edits': 0}

def estimate_common_crawl_proportion(lang_code):
    """估计Common Crawl中的比例（简化示例）"""
    # 实际实现需要分析Common Crawl语言识别数据
    # 这里使用预设值
    cc_estimates = {
        'en': 60.0, 'zh-CN': 15.0, 'es': 5.0, 'ar': 3.0,
        'hi': 3.0, 'bn': 2.0, 'pt': 2.0, 'ru': 2.0,
        'ja': 2.0, 'de': 2.0, 'fr': 2.0, 'it': 1.5,
        'uk': 0.5, 'th': 0.5, 'he': 0.3, 'zu': 0.05,
        'gd': 0.02, 'hmn': 0.01, 'gn': 0.005
    }

    return cc_estimates.get(lang_code, 0.01)

def get_opus_corpus_size(lang_code):
    """获取OPUS语料库大小（简化示例）"""
    # OPUS语料大小映射（单位：百万词）
    opus_sizes = {
        'en': 5000, 'zh-CN': 800, 'ar': 300, 'hi': 200,
        'bn': 150, 'uk': 100, 'th': 80, 'he': 60,
        'it': 400, 'zu': 5, 'gd': 3, 'hmn': 2, 'gn': 1
    }

    return opus_sizes.get(lang_code, 1)

def collect_data_sources(lang_code):
    """收集多源数据"""
    sources = {}

    # 1. Wikipedia数据
    wiki_stats = get_wikipedia_stats(lang_code)
    sources['wikipedia_articles'] = wiki_stats['article_count']
    sources['wikipedia_bytes'] = wiki_stats['total_bytes']

    # 2. Common Crawl估计
    sources['common_crawl_proportion'] = estimate_common_crawl_proportion(lang_code)

    # 3. OPUS语料大小
    sources['opus_size_mb'] = get_opus_corpus_size(lang_code)

    # 4. 文献估计
    literature_estimates = {
        'en': 60.0, 'zh-CN': 15.0, 'ar': 3.0, 'hi': 3.0,
        'bn': 2.0, 'it': 1.5, 'uk': 0.5, 'th': 0.5,
        'he': 0.3, 'zu': 0.05, 'gd': 0.02, 'hmn': 0.01, 'gn': 0.005
    }
    sources['literature_estimate'] = literature_estimates.get(lang_code, 0.01)

    return sources

def estimate_proportion(lang_code, sources):
    """加权平均估算比例"""
    # 权重配置（可根据数据质量调整）
    weights = {
        'wikipedia_articles': 0.25,
        'common_crawl_proportion': 0.35,
        'opus_size_mb': 0.20,
        'literature_estimate': 0.20
    }

    # 归一化处理（对数尺度处理数量级差异）
    normalized = {}
    for key, value in sources.items():
        if 'proportion' in key:
            normalized[key] = value / 100  # 转换为比例
        else:
            # 对数归一化
            normalized[key] = log(1 + value) if value > 0 else 0

    # 加权平均
    weighted_sum = 0
    total_weight = 0
    for key, weight in weights.items():
        if key in normalized:
            weighted_sum += normalized[key] * weight
            total_weight += weight

    estimated = weighted_sum / total_weight if total_weight > 0 else 0

    # 转换为百分比（相对英语，假设英语占比60%）
    english_proportion = 0.60  # 60%
    relative_proportion = (estimated / english_proportion) * 100

    return {
        'estimated_proportion': round(relative_proportion, 4),
        'confidence_low': round(relative_proportion * 0.7, 4),
        'confidence_high': round(relative_proportion * 1.3, 4),
        'source_breakdown': sources
    }

# 批量处理所有语言
languages = ['zu', 'gd', 'hmn', 'gn', 'uk', 'bn', 'th', 'he', 'zh-CN', 'ar', 'it', 'hi']

results = []
for lang in languages:
    print(f"Processing {lang}...")
    sources = collect_data_sources(lang)
    estimate = estimate_proportion(lang, sources)
    estimate['language'] = lang
    results.append(estimate)

# 保存结果
df = pd.DataFrame(results)
df.to_csv('data/processed/training_proportion_features.csv', index=False)
print(f"Saved training proportion estimates for {len(df)} languages")
```

### 特征变量命名
- `training_data_proportion`: 预估占比 (百分比)
- `proportion_confidence_low`: 置信区间下限
- `proportion_confidence_high`: 置信区间上限
- `proportion_sources`: 数据源详情 (JSON格式)

### 验证命令
```bash
# 运行比例估算
python scripts/estimate_training_proportion.py

# 查看估计结果
python -c "import pandas as pd; df=pd.read_csv('data/processed/training_proportion_features.csv'); print(df[['language', 'estimated_proportion']].sort_values('estimated_proportion', ascending=False))"
```

---

## 5. 音系复杂度

### 定义
音位库存大小和音系规则复杂度，包括辅音、元音数量，是否有声调等特征。

### 数据来源
1. **PHOIBLE数据库** - 音位库存数据
2. **UPSID数据库** - UCLA音位库存数据库
3. **语言学文献** - 特定语言音系研究

### 执行步骤

#### 步骤1: PHOIBLE数据获取
```bash
# 安装phoible Python包（如果可用）
pip install phoible

# 或直接下载PHOIBLE CSV数据
wget https://github.com/phoible/dev/raw/master/data/phoible.csv -O data/raw/phoible.csv
```

#### 步骤2: 音系复杂度分析脚本
创建 `scripts/analyze_phonology.py`:

```python
import pandas as pd
import numpy as np

def get_phonological_complexity(lang_code):
    """从PHOIBLE获取音系数据并计算复杂度"""
    # 加载PHOIBLE数据
    try:
        phoible_df = pd.read_csv('data/raw/phoible.csv')
    except FileNotFoundError:
        print("PHOIBLE data not found. Using manual annotations.")
        return get_manual_phonology(lang_code)

    # 语言代码映射（PHOIBLE使用ISO 639-3代码）
    iso_mapping = {
        'zu': 'zul', 'gd': 'gla', 'hmn': 'hmn', 'gn': 'grn',
        'uk': 'ukr', 'bn': 'ben', 'th': 'tha', 'he': 'heb',
        'zh-CN': 'cmn', 'ar': 'arb', 'it': 'ita', 'hi': 'hin'
    }

    iso_code = iso_mapping.get(lang_code, lang_code)

    # 筛选指定语言
    lang_data = phoible_df[phoible_df['ISO6393'] == iso_code]

    if lang_data.empty:
        # 备用：使用语言名称匹配
        lang_name_mapping = {
            'zu': 'Zulu', 'gd': 'Scottish Gaelic', 'hmn': 'Hmong', 'gn': 'Guarani',
            'uk': 'Ukrainian', 'bn': 'Bengali', 'th': 'Thai', 'he': 'Hebrew',
            'zh-CN': 'Mandarin Chinese', 'ar': 'Arabic', 'it': 'Italian', 'hi': 'Hindi'
        }
        lang_name = lang_name_mapping.get(lang_code, '')
        lang_data = phoible_df[phoible_df['Name'].str.contains(lang_name, na=False, case=False)]

    if not lang_data.empty:
        # 提取音位信息（取第一个方言的数据）
        row = lang_data.iloc[0]

        # 辅音和元音数量
        consonants = int(row['Consonants']) if 'Consonants' in row and pd.notna(row['Consonants']) else 0
        vowels = int(row['Vowels']) if 'Vowels' in row and pd.notna(row['Vowels']) else 0
        tones = int(row['Tones']) if 'Tones' in row and pd.notna(row['Tones']) else 0

        total_phonemes = consonants + vowels

        # 复杂度评分
        complexity_score = 0
        complexity_score += min(total_phonemes / 50, 1.0)  # 音位数量（上限1.0）
        complexity_score += 0.3 if tones > 0 else 0        # 声调系统
        complexity_score += 0.2 if 'click' in str(row.get('SpecialFeatures', '')) else 0  # 特殊音位
        complexity_score += 0.1 if 'implosive' in str(row.get('SpecialFeatures', '')) else 0

        # 归一化到0-2范围
        final_score = round(min(complexity_score, 2.0), 3)

        return {
            'consonant_count': consonants,
            'vowel_count': vowels,
            'total_phonemes': total_phonemes,
            'has_tones': tones > 0,
            'tone_count': tones if tones > 0 else 0,
            'phonological_complexity': final_score,
            'source': 'PHOIBLE'
        }

    # PHOIBLE中没有找到，使用手动标注
    return get_manual_phonology(lang_code)

def get_manual_phonology(lang_code):
    """手动标注的音系数据"""
    manual_data = {
        'zu': {
            'consonant_count': 45,
            'vowel_count': 10,
            'has_tones': False,
            'tone_count': 0,
            'complexity_score': 1.2,
            'notes': '丰富的辅音系统，包含吸气音和搭嘴音'
        },
        'zh-CN': {
            'consonant_count': 22,
            'vowel_count': 9,
            'has_tones': True,
            'tone_count': 4,
            'complexity_score': 1.8,
            'notes': '有声调系统，增加复杂度'
        },
        'th': {
            'consonant_count': 44,
            'vowel_count': 32,
            'has_tones': True,
            'tone_count': 5,
            'complexity_score': 2.0,
            'notes': '复杂的声调系统和元音系统'
        },
        'gd': {
            'consonant_count': 27,
            'vowel_count': 14,
            'has_tones': False,
            'tone_count': 0,
            'complexity_score': 0.9,
            'notes': '凯尔特语系，有一定音系复杂性'
        },
        # ... 其他语言的预设数据
    }

    data = manual_data.get(lang_code, {
        'consonant_count': None,
        'vowel_count': None,
        'has_tones': None,
        'tone_count': None,
        'complexity_score': None,
        'notes': 'No data available'
    })

    return {
        'consonant_count': data['consonant_count'],
        'vowel_count': data['vowel_count'],
        'total_phonemes': (data['consonant_count'] or 0) + (data['vowel_count'] or 0),
        'has_tones': data['has_tones'],
        'tone_count': data['tone_count'],
        'phonological_complexity': data['complexity_score'],
        'source': 'Manual'
    }

# 批量处理所有语言
languages = ['zu', 'gd', 'hmn', 'gn', 'uk', 'bn', 'th', 'he', 'zh-CN', 'ar', 'it', 'hi']

results = []
for lang in languages:
    print(f"Processing {lang}...")
    phonology = get_phonological_complexity(lang)
    phonology['language'] = lang
    results.append(phonology)

# 保存结果
df = pd.DataFrame(results)
df.to_csv('data/processed/phonology_features.csv', index=False)
print(f"Saved phonology features for {len(df)} languages")
```

### 特征变量命名
- `consonant_count`: 辅音数量
- `vowel_count`: 元音数量
- `total_phonemes`: 音位总数 (辅音+元音)
- `has_tones`: 是否有声调 (布尔)
- `tone_count`: 声调数量 (如有)
- `phonological_complexity`: 复杂度得分 (0-2)

### 验证命令
```bash
# 运行音系分析
python scripts/analyze_phonology.py

# 查看音系复杂度分布
python -c "import pandas as pd; df=pd.read_csv('data/processed/phonology_features.csv'); print(df[['language', 'total_phonemes', 'has_tones', 'phonological_complexity']].sort_values('phonological_complexity', ascending=False))"
```

---

## 6. 书写系统

### 定义
文字类型、书写方向和相关特征。

### 数据来源
1. **Unicode标准** - 文字区块和属性
2. **ISO 15924标准** - 文字代码
3. **语言学文献** - 文字系统描述

### 执行步骤

#### 步骤1: 文字系统映射脚本
创建 `scripts/analyze_writing_system.py`:

```python
# 语言到文字代码的映射（ISO 15924）
SCRIPT_MAPPING = {
    'zu': 'Latn',  # 拉丁字母
    'gd': 'Latn',
    'hmn': 'Latn',  # 苗文（拉丁变体）
    'gn': 'Latn',
    'uk': 'Cyrl',  # 西里尔字母
    'bn': 'Beng',  # 孟加拉文
    'th': 'Thai',  # 泰文
    'he': 'Hebr',  # 希伯来文
    'zh-CN': 'Hans',  # 简体中文
    'ar': 'Arab',  # 阿拉伯文
    'it': 'Latn',
    'hi': 'Deva'   # 天城文
}

# 文字属性定义
SCRIPT_PROPERTIES = {
    'Latn': {
        'type': 'alphabet',
        'direction': 'LTR',
        'has_case': True,
        'has_ligatures': False,
        'unicode_blocks': ['Basic Latin', 'Latin-1 Supplement', 'Latin Extended-A'],
        'complexity_score': 0.8
    },
    'Cyrl': {
        'type': 'alphabet',
        'direction': 'LTR',
        'has_case': True,
        'has_ligatures': False,
        'unicode_blocks': ['Cyrillic', 'Cyrillic Supplement'],
        'complexity_score': 1.0
    },
    'Beng': {
        'type': 'abugida',
        'direction': 'LTR',
        'has_case': False,
        'has_ligatures': True,
        'unicode_blocks': ['Bengali'],
        'complexity_score': 1.5
    },
    'Thai': {
        'type': 'abugida',
        'direction': 'LTR',
        'has_case': False,
        'has_ligatures': True,
        'unicode_blocks': ['Thai'],
        'complexity_score': 1.6
    },
    'Hebr': {
        'type': 'abjad',
        'direction': 'RTL',
        'has_case': False,
        'has_ligatures': False,
        'unicode_blocks': ['Hebrew'],
        'complexity_score': 1.2
    },
    'Hans': {
        'type': 'logographic',
        'direction': 'LTR',
        'has_case': False,
        'has_ligatures': False,
        'unicode_blocks': ['CJK Unified Ideographs', 'CJK Symbols and Punctuation'],
        'complexity_score': 2.0
    },
    'Arab': {
        'type': 'abjad',
        'direction': 'RTL',
        'has_case': False,
        'has_ligatures': True,
        'unicode_blocks': ['Arabic', 'Arabic Supplement'],
        'complexity_score': 1.8
    },
    'Deva': {
        'type': 'abugida',
        'direction': 'LTR',
        'has_case': False,
        'has_ligatures': True,
        'unicode_blocks': ['Devanagari'],
        'complexity_score': 1.7
    }
}

def analyze_writing_system(lang_code):
    """分析语言的书写系统"""
    script_code = SCRIPT_MAPPING.get(lang_code, 'Unknown')

    if script_code == 'Unknown':
        return {
            'script_code': 'Unknown',
            'script_type': 'unknown',
            'direction': 'unknown',
            'has_case': None,
            'has_ligatures': None,
            'script_complexity': None,
            'source': 'Unknown'
        }

    props = SCRIPT_PROPERTIES.get(script_code, {})

    # 文字类型名称映射
    type_names = {
        'alphabet': '字母文字',
        'abugida': '元音附标文字',
        'abjad': '辅音音素文字',
        'logographic': '意音文字',
        'syllabary': '音节文字'
    }

    return {
        'script_code': script_code,
        'script_type': props.get('type', 'unknown'),
        'script_type_chinese': type_names.get(props.get('type', ''), '未知'),
        'direction': props.get('direction', 'unknown'),
        'has_case': props.get('has_case', False),
        'has_ligatures': props.get('has_ligatures', False),
        'script_complexity': props.get('complexity_score', 0),
        'unicode_blocks': ', '.join(props.get('unicode_blocks', [])),
        'source': 'ISO 15924 / Unicode'
    }

def validate_writing_system(lang_code, sample_text):
    """通过样本文本验证书写系统"""
    import re

    # 检测文字方向（简化）
    direction = 'unknown'
    if sample_text:
        # 简单启发式：检查首字符
        first_char = sample_text[0] if sample_text else ''
        # RTL文字通常在Unicode的特定范围
        rtl_ranges = [
            (0x0590, 0x05FF),  # Hebrew
            (0x0600, 0x06FF),  # Arabic
            (0x0700, 0x074F),  # Syriac
            # ... 其他RTL文字范围
        ]

        char_code = ord(first_char)
        if any(start <= char_code <= end for start, end in rtl_ranges):
            direction = 'RTL'
        else:
            direction = 'LTR'

    # 检测文字类型特征
    detected_scripts = {
        'latin': bool(re.search(r'[A-Za-z]', sample_text)),
        'cyrillic': bool(re.search(r'[А-Яа-я]', sample_text)),
        'chinese': bool(re.search(r'[\u4e00-\u9fff]', sample_text)),
        'arabic': bool(re.search(r'[\u0600-\u06FF]', sample_text)),
        'devanagari': bool(re.search(r'[\u0900-\u097F]', sample_text)),
        'bengali': bool(re.search(r'[\u0980-\u09FF]', sample_text)),
        'thai': bool(re.search(r'[\u0E00-\u0E7F]', sample_text)),
        'hebrew': bool(re.search(r'[\u0590-\u05FF]', sample_text))
    }

    return {
        'detected_direction': direction,
        'detected_scripts': {k: v for k, v in detected_scripts.items() if v}
    }

# 批量处理所有语言
languages = ['zu', 'gd', 'hmn', 'gn', 'uk', 'bn', 'th', 'he', 'zh-CN', 'ar', 'it', 'hi']

results = []
for lang in languages:
    print(f"Processing {lang}...")
    writing_system = analyze_writing_system(lang)
    writing_system['language'] = lang
    results.append(writing_system)

# 保存结果
import pandas as pd
df = pd.DataFrame(results)
df.to_csv('data/processed/writing_system_features.csv', index=False)
print(f"Saved writing system features for {len(df)} languages")
```

### 特征变量命名
- `script_type`: 文字类型 (alphabet/abugida/abjad/logographic/syllabary)
- `writing_direction`: 书写方向 (LTR/RTL/TTB/BTT)
- `has_case_system`: 是否有大小写 (布尔)
- `has_ligatures`: 是否有连字 (布尔)
- `script_complexity`: 文字复杂度评分 (0-2)

### 验证命令
```bash
# 运行书写系统分析
python scripts/analyze_writing_system.py

# 查看文字类型分布
python -c "import pandas as pd; df=pd.read_csv('data/processed/writing_system_features.csv'); print(df[['language', 'script_type', 'direction', 'script_complexity']])"
```

---

## 完整执行流程

### 阶段1: 环境设置
```bash
# 创建项目目录结构
mkdir -p {data/{raw,processed,external},scripts,config,outputs,notebooks}

# 创建Python虚拟环境
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# 或 .venv\Scripts\activate  # Windows

# 安装基础依赖
pip install pandas numpy requests beautifulsoup4 lxml scipy scikit-learn pyyaml
```

### 阶段2: 配置文件创建
创建 `config/languages.yaml`:
```yaml
languages:
  - code: zu
    name: Zulu
    family: Niger-Congo
    resource_level: LRL
  - code: gd
    name: Scots Gaelic
    family: Indo-European
    resource_level: LRL
  - code: hmn
    name: Hmong
    family: Hmong-Mien
    resource_level: LRL
  # ... 其他语言配置
```

### 阶段3: 批量特征提取
创建 `scripts/collect_all_features.py` 主脚本，协调所有特征提取过程。

### 阶段4: 质量控制和验证
1. 人工抽查验证 (10%样本)
2. 特征一致性检查
3. 缺失值处理
4. 生成质量报告

### 阶段5: 特征合并与分析
合并所有特征到单一数据集，进行描述性统计和相关性分析。

---

## 特征变量汇总表

| 特征类别 | 变量名 | 数据类型 | 取值范围 | 描述 |
|---------|--------|----------|----------|------|
| 形态复杂度 | `morph_type` | 分类 | Isolating/Fusional/Agglutinative | 形态类型 |
| 词序灵活度 | `basic_word_order` | 分类 | SVO/SOV/VSO/VOS/OVS/OSV | 基本词序 |
| | `order_flexibility` | 数值 | 0-1 | 词序灵活度得分 |
| 语法距离 | `grammar_distance_composite` | 数值 | 0-1 | 与英语的语法距离 |
| 训练数据占比 | `training_data_proportion` | 数值 | 0-100 | 预估训练数据占比 |
| 音系复杂度 | `phonological_complexity` | 数值 | 0-2 | 音系复杂度得分 |
| 书写系统 | `script_type` | 分类 | alphabet/abugida/abjad/logographic | 文字类型 |
| | `script_complexity` | 数值 | 0-2 | 文字复杂度得分 |

---

## 预期产出文件

1. **特征数据文件**: `data/processed/language_features.csv`
   - 包含12种语言的6个特征完整数据
2. **特征提取脚本**: `scripts/extract_*.py` (6个脚本)
3. **配置文件**: `config/languages.yaml`, `config/grammar_features.yaml`
4. **分析报告**: `outputs/feature_analysis_report.md`
5. **可视化图表**: `outputs/feature_distributions/` 目录

---

## 注意事项

1. **数据可用性**: 低资源语言数据可能有限，需要准备手动标注方案
2. **API限制**: WALS等API可能有请求限制，需实现缓存和重试机制
3. **特征歧义**: 某些语言特征可能模糊，应记录置信度和注释
4. **更新维护**: 语言特征可能随研究进展更新，应设计可扩展的标注系统
