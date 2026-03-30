# TASK.md - 任务列表

## 执行顺序与依赖关系

```
Phase A (基础设施)
  [T1] config/cot_config.yaml ──┐
  [T2] resp_model_config.yaml ──┼──> [T3] cot_extraction.py
                                │
Phase B (数据收集)              │
  运行 cot_extraction.py ───────┘
                                │
Phase C (标注)                  │
  [T4] auto_annotator.py <──────┘
                                │
Phase D (分析)                  │
  [T5] cot_truncation_analysis.py <──┤
  [T6] cot_comparison.py <──────────┘
```

---

## Phase A: 基础设施搭建

### [T1] 创建 config/cot_config.yaml
- **状态**: pending
- **描述**: 定义 CoT 实验参数
  - `thinking_budgets`: [512, 1024, 2048, 4096]
  - `pilot_languages`: [en, zu, gd, it]
  - `all_languages`: 全部 12 种
  - `prompt_sources`: [isc_bench, advbench]
  - `advbench_sample_size`: 20

### [T2] 修改 config/resp_model_config.yaml
- **状态**: pending
- **描述**: 取消注释 DeepSeek-chat 配置，设置 `thinking: true`，`thinking_budget_tokens: 1024`

### [T3] 创建 scripts/cot_extraction.py
- **状态**: pending
- **依赖**: T1, T2
- **描述**: CoT 提取主脚本
  - 基于 `extract_advbench_output.py` 框架
  - 支持 ISC-Bench prompt (`--prompt-type isc_bench`) 和 AdvBench (`--prompt-type advbench`)
  - 捕获 `reasoning_content` (DeepSeek) 或解析 `<think>` 标签
  - 记录 thinking_tokens, resp_tokens
  - 支持 `--budget`, `--lang`, `--limit` 参数
  - 输出到 `outputs/cot/{model_id}/cot_raw_{model_id}.csv`

---

## Phase C: CoT 标注

### [T4] 创建 scripts/annotation/auto_annotator.py
- **状态**: pending
- **依赖**: T3（需要先有 CoT 数据）
- **描述**: LLM 自动标注脚本
  - 读取 `cot_raw_{model_id}.csv`
  - 使用 DeepSeek-chat（非 thinking mode）逐条标注 CoT 阶段
  - 阶段体系: LANG_RECOGNITION, TASK_PARSING, SAFETY_EVALUATION, COMPLIANCE_DECISION, CONTENT_PLANNING, SELF_CENSORSHIP, OTHER
  - 输出 JSON 格式的阶段标注
  - 计算 stage_count, has_safety_stage, safety_stage_ratio, cot_language, is_truncated
  - 写入 `cot_annotated_{model_id}.csv`

---

## Phase D: 分析与可视化

### [T5] 创建 scripts/cot_truncation_analysis.py
- **状态**: pending
- **依赖**: T3
- **描述**: CoT 截断检测与统计
  - 各语言/资源水平的 CoT 长度分布箱线图
  - Mann-Whitney U 统计检验（LRL vs HRL）
  - 截断率计算（绝对截断 + 相对截断 + 阶段缺失）
  - 不同 budget 下的截断率对比
  - 输出图表到 `outputs/cot/analysis/figures/`

### [T6] 创建 scripts/analysis/cot_comparison.py
- **状态**: pending
- **依赖**: T4, T5
- **描述**: ISC-Bench vs AdvBench 对比分析
  - 同一语言下 CoT 安全推理触发率对比
  - SAFETY_EVALUATION 阶段的深度/长度对比
  - COMPLIANCE_DECISION 倾向性差异
  - 生成对比图表和统计报告
