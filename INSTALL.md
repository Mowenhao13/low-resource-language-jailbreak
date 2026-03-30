# 语言特征标注系统 - 安装与使用指南

## 项目概述

本项目基于论文 **"Low-Resource Languages Jailbreak GPT-4"** 的研究，为12种语言标注6个关键语言特征，用于分析语言特征与越狱成功率(ASR)的相关性。

### 语言列表（12种）
- **低资源语言**: 祖鲁语(zu)、苏格兰盖尔语(gd)、苗语(hmn)、瓜拉尼语(gn)
- **中资源语言**: 乌克兰语(uk)、孟加拉语(bn)、泰语(th)、希伯来语(he)
- **高资源语言**: 简体中文(zh-CN)、阿拉伯语(ar)、意大利语(it)、印地语(hi)
zu gd hmn gn uk bn th he zh-CN ar it hi

### 标注特征（6个）
1. **形态复杂度** - 屈折/孤立/黏着语类型
2. **词序灵活度** - 基本词序和灵活程度
3. **语法距离** - 与英语的语法相似度
4. **训练数据占比** - 在LLM训练数据中的预估比例
5. **音系复杂度** - 音位库存和复杂度
6. **书写系统** - 文字类型和书写方向

## 快速开始

### 1. 环境设置

```bash
# 克隆或进入项目目录
cd /Users/halllo/projects/local/low-resource-lang-jailbreak

# 创建Python虚拟环境（推荐）
python -m venv .venv

# 激活虚拟环境
# Linux/macOS:
source .venv/bin/activate
# Windows:
# .venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

### 2. 项目结构初始化

项目已包含以下结构：
```
low-resource-lang-jailbreak/
├── data/                    # 数据目录
│   ├── raw/                # 原始数据（从外部源下载）
│   └── processed/          # 处理后的特征数据
├── scripts/                # Python脚本
│   └── collect_all_features.py  # 主协调脚本
├── config/                 # 配置文件
│   ├── languages.yaml     # 语言配置
│   └── grammar_features.yaml # 语法特征配置
├── outputs/               # 输出报告和图表
├── task1-语言特征标注.md  # 详细执行方案文档
├── requirements.txt       # Python依赖
└── INSTALL.md            # 本文件
```

### 3. 运行特征提取

#### 测试运行（检查环境）
```bash
python scripts/collect_all_features.py --check-only
```

#### 完整特征提取
```bash
# 提取所有特征（当前为框架版本）
python scripts/collect_all_features.py

# 提取指定特征
python scripts/collect_all_features.py --features morphology word_order

# 指定输出文件
python scripts/collect_all_features.py --output data/processed/my_features.csv
```

### 4. 实现具体特征提取

当前主脚本为框架版本，各个特征提取函数需要具体实现：

1. **形态复杂度**: 实现 `scripts/extract_morphology.py`
2. **词序灵活度**: 实现 `scripts/analyze_word_order.py`
3. **语法距离**: 实现 `scripts/calculate_grammar_distance.py`
4. **训练数据占比**: 实现 `scripts/estimate_training_proportion.py`
5. **音系复杂度**: 实现 `scripts/analyze_phonology.py`
6. **书写系统**: 实现 `scripts/analyze_writing_system.py`

每个脚本的详细实现方案见 `task1-语言特征标注.md`。

## 详细执行步骤

### 阶段1: 数据收集准备
```bash
# 创建必要的目录
mkdir -p data/{raw,processed,external} scripts config outputs notebooks

# 下载必要的数据集（示例）
# wget https://wals.info/feature/26A.csv -O data/raw/wals_morphology.csv
# wget https://wals.info/feature/82A.csv -O data/raw/wals_word_order.csv
```

### 阶段2: 配置检查
```bash
# 检查配置文件
cat config/languages.yaml
cat config/grammar_features.yaml
```

### 阶段3: 逐个实现特征提取模块
参考 `task1-语言特征标注.md` 中的详细实现方案，逐个完成6个特征提取脚本。

### 阶段4: 集成测试
```bash
# 测试单个特征模块
python scripts/extract_morphology.py

# 运行完整流程
python scripts/collect_all_features.py --output data/processed/final_features.csv
```

### 阶段5: 质量控制和验证
1. 检查输出数据的完整性
2. 验证特征值的合理性
3. 生成质量报告

## 数据源说明

### 主要数据源
1. **WALS数据库**: 世界语言结构图谱，用于形态、词序、语法特征
   - 网址: https://wals.info/
   - API文档: https://wals.info/api

2. **PHOIBLE数据库**: 音位库存数据，用于音系复杂度
   - 网址: https://phoible.org/
   - 数据: https://github.com/phoible/dev

3. **Ethnologue**: 语言分类和元数据
   - 网址: https://www.ethnologue.com/

4. **Wikipedia统计**: 各语言版本规模，用于训练数据占比估计
   - API: https://{lang}.wikipedia.org/w/api.php

5. **Common Crawl**: 网页语言分布统计
   - 网址: https://commoncrawl.org/

### 备用数据源
- **Glottolog**: 语言谱系数据库
- **ASJP**: 自动化相似度判断程序
- **OPUS**: 多语言平行语料库
- **Unicode标准**: 文字系统和属性

## 特征变量命名规范

### 通用规则
- 使用小写字母和下划线 (`snake_case`)
- 前缀表示特征类别，如 `morph_`, `phon_`, `script_`
- 后缀表示数据类型，如 `_type`, `_score`, `_count`, `_confidence`

### 具体变量名
1. **形态复杂度**: `morph_type`, `morph_confidence`, `morph_source`
2. **词序灵活度**: `basic_word_order`, `order_flexibility`, `order_entropy`
3. **语法距离**: `grammar_distance_jaccard`, `grammar_distance_hamming`, `grammar_distance_composite`
4. **训练数据占比**: `training_data_proportion`, `proportion_confidence_low`, `proportion_confidence_high`
5. **音系复杂度**: `consonant_count`, `vowel_count`, `total_phonemes`, `has_tones`, `phonological_complexity`
6. **书写系统**: `script_type`, `writing_direction`, `has_case_system`, `has_ligatures`, `script_complexity`

## 常见问题

### Q1: WALS API访问限制
WALS API可能有请求频率限制，建议：
- 实现请求缓存（将结果保存到本地文件）
- 添加适当的延迟（如 `time.sleep(1)` 在请求之间）
- 使用离线数据集（可下载完整的WALS数据）

### Q2: 低资源语言数据缺失
对于WALS等数据库中缺失的低资源语言数据：
1. 使用备用数据源（Ethnologue, Glottolog）
2. 参考语言学文献进行手动标注
3. 使用相似语言的代理数据

### Q3: 数据不一致性
不同数据源可能对同一语言提供不同的特征值：
1. 记录数据来源 (`source` 字段)
2. 提供置信度评分 (`confidence` 字段)
3. 支持多值标注（如主要类型和次要类型）

### Q4: 实现复杂度
如果时间有限，建议优先实现：
1. 书写系统（最简单，基于映射表）
2. 音系复杂度（PHOIBLE数据相对规整）
3. 形态复杂度（WALS特征26A）
4. 其他特征按需实现

## 扩展与定制

### 添加新语言
在 `config/languages.yaml` 中添加新的语言条目，包括：
- ISO语言代码
- 语言名称（英文和中文）
- 语系和语族
- 资源水平分类（LRL/MRL/HRL）

### 添加新特征
1. 在配置文件中定义特征参数
2. 实现特征提取脚本
3. 更新主协调脚本 `collect_all_features.py`
4. 更新特征变量命名规范

### 修改特征权重
在 `config/grammar_features.yaml` 中调整 `feature_weights` 部分，用于加权距离计算。

## 联系方式与支持

### 文档
- 详细执行方案: `task1-语言特征标注.md`
- 论文参考: `2310.02446v2.pdf`
- 项目说明: `CLAUDE.md`

### 故障排除
1. 检查Python依赖是否安装正确
2. 验证配置文件格式（YAML语法）
3. 检查网络连接（API访问需要）
4. 查看脚本错误日志

### 后续开发
完成特征提取后，下一步工作：
1. 与越狱成功率数据进行关联分析
2. 建立回归模型分析特征重要性
3. 识别高漏洞语言特征组合
4. 构建多语言安全评估框架

---
*最后更新: 2026-03-28*