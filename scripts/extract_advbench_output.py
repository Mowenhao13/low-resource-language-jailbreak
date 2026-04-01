import pandas as pd
import csv
import sys
import yaml
import json
import argparse
from pathlib import Path
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# Add scripts directory to path for translator import
sys.path.insert(0, str(Path(__file__).parent))
from translate.nllb_translator import NLLBTranslator, get_nllb_lang_code
from eval.evalutor import get_scores

# Get project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Language list
LANGS = ['zu', 'gd', 'xh', 'gn', 'uk', 'bn', 'th', 'he', 'zh-CN', 'ar', 'it', 'hi']


def load_resp_config():
    """Load response model configuration (Stage 1)"""
    config_path = PROJECT_ROOT / 'config' / 'resp_model_config.yaml'
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_translator_config():
    """Load translator configuration (Stage 2)"""
    config_path = PROJECT_ROOT / 'config' / 'translator_config.yaml'
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_eval_config():
    """Load evaluation model configuration (Stage 3)"""
    config_path = PROJECT_ROOT / 'config' / 'eval_model_config.yaml'
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def create_llm_client(config):
    """Create OpenAI client from model config"""
    client = OpenAI(
        base_url=config.get('base_url'),
        api_key=config.get('api_key') or 'dummy-key'
    )
    return client


def query_llm_with_instruction(client, model_cfg, instruction):
    """
    使用翻译后的指令查询 LLM

    Args:
        client: OpenAI 客户端
        model_cfg: 模型配置
        instruction: 翻译后的指令

    Returns:
        LLM 的响应文本
    """
    if not model_cfg.get('api_key'):
        return f"[Placeholder response from {model_cfg.get('model_id')}]"

    # Build API parameters
    api_params = {
        'model': model_cfg.get('model_id'),
        'messages': [{"role": "user", "content": instruction}],
        'max_tokens': model_cfg.get('max_tokens', 2048),
        'temperature': 0
    }

    # Add thinking mode parameters if enabled
    if model_cfg.get('thinking', False):
        # For DeepSeek-style thinking (uses extra_body)
        api_params['extra_body'] = {
            'thinking': {
                'type': 'enabled',
                'budget_tokens': model_cfg.get('thinking_budget_tokens', 1024)
            }
        }

    completion = client.chat.completions.create(**api_params)
    model_resp = completion.choices[0].message.content

    # If content is empty but there's reasoning_content, use that as fallback (DeepSeek)
    if not model_resp and hasattr(completion.choices[0].message, 'reasoning_content'):
        model_resp = completion.choices[0].message.reasoning_content

    return model_resp


def translate_llm_output(translator, model_resp, source_lang_code, target_lang_code='en'):
    """
    翻译 LLM 输出回英文

    Args:
        translator: NLLBTranslator 实例
        model_resp: LLM 的原始响应
        source_lang_code: 源语言代码
        target_lang_code: 目标语言代码 (默认: en)

    Returns:
        翻译后的文本
    """
    try:
        if model_resp and not model_resp.startswith('['):
            return translator.translate(
                model_resp,
                source_lang=get_nllb_lang_code(source_lang_code),
                target_lang=get_nllb_lang_code(target_lang_code)
            )
        else:
            return model_resp
    except Exception as e:
        print(f"  Translation error: {e}")
        return f"[Translation failed: {str(e)}]"


def evaluate_response(eval_client, eval_model_cfg, source_text, model_resp):
    """
    使用评估模型评估 LLM 响应

    Args:
        eval_client: 评估模型客户端
        eval_model_cfg: 评估模型配置
        source_text: 原始英文有害指令
        model_resp: 翻译后的 LLM 响应

    Returns:
        float: 评估综合得分
    """

    thinking_mode = False 
    # Add thinking mode parameters if enabled for evaluation
    if eval_model_cfg.get('thinking', True):
        thinking_mode = True 

    return get_scores(source_text, model_resp, eval_client, eval_model_cfg.get('model_id'), thinking_mode)

    


def get_processed_rows(output_csv):
    """获取已处理的行集合，用于断点续传"""
    processed = set()
    if output_csv.exists():
        try:
            df = pd.read_csv(output_csv)
            for _, row in df.iterrows():
                key = (row['source_lang_code'], row['source_text'])
                processed.add(key)
        except Exception:
            pass
    return processed


def process_single_item(item):
    """处理单个指令（并发执行的单元）"""
    lang, instruction, source_text, resp_clients = item
    results = {}

    for model_id, model_data in resp_clients.items():
        client = model_data['client']
        model_cfg = model_data['config']

        try:
            model_resp = query_llm_with_instruction(client, model_cfg, instruction)
        except Exception as e:
            print(f"  Error with {model_id}: {e}")
            model_resp = f"[Error: {str(e)}]"

        results[model_id] = (lang, instruction, source_text, model_resp)

    return results


def run_stage1(limit=None, max_workers=5):
    """
    Stage 1: 使用翻译后的指令查询 LLM，将原始输出写入 CSV
    支持断点续传和并发执行

    CSV 结构: source_lang_code, input_text, source_text, model_resp
    """
    print("=" * 80)
    print("STAGE 1: 查询 LLM 获取原始响应")
    print("=" * 80)

    # Load response model config
    resp_config = load_resp_config()

    # Setup response model clients and output directories
    resp_clients = {}
    resp_output_dirs = {}
    llm_output_base = PROJECT_ROOT / 'outputs' / 'llm'
    llm_output_base.mkdir(parents=True, exist_ok=True)

    for model_cfg in resp_config.get('models', []):
        model_id = model_cfg['model_id']
        resp_clients[model_id] = {
            'client': create_llm_client(model_cfg),
            'config': model_cfg
        }
        model_dir = llm_output_base / model_id
        model_dir.mkdir(exist_ok=True)
        resp_output_dirs[model_id] = model_dir

    # Read input data
    input_df = pd.read_csv(PROJECT_ROOT / 'data' / 'processed' / 'advbench_input.csv')

    print(f"\nharmful_behaviors: {len(input_df)} 条指令")
    if limit:
        print(f"Testing with limit: {limit} instructions per language")

    # 为每个模型加载已处理的行
    processed_sets = {}
    output_csvs = {}
    for model_id in resp_clients.keys():
        output_csv = resp_output_dirs[model_id] / f'advbench_output_{model_id}_stage1.csv'
        output_csvs[model_id] = output_csv
        processed_sets[model_id] = get_processed_rows(output_csv)
        print(f"  {model_id}: 已处理 {len(processed_sets[model_id])} 行")

    # Setup CSV writers for each response model (追加模式)
    csv_writers = {}
    csv_files = {}
    file_locks = {}

    for model_id in resp_clients.keys():
        output_csv = output_csvs[model_id]
        file_exists = output_csv.exists()
        f = open(output_csv, 'a', encoding='utf-8', newline='')
        csv_files[model_id] = f
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['source_lang_code', 'input_text', 'source_text', 'model_resp'])
        csv_writers[model_id] = writer
        file_locks[model_id] = Lock()

    # 构建待处理任务列表
    tasks = []
    for lang in LANGS:
        lang_df = input_df[input_df['target_lang_code'] == lang]
        if limit:
            lang_df = lang_df.head(limit)

        for _, row in lang_df.iterrows():
            instruction = row['translated_text']
            source_text = row['source_text']
            key = (lang, source_text)

            # 检查是否所有模型都已处理这一行
            all_processed = True
            for model_id in resp_clients.keys():
                if key not in processed_sets[model_id]:
                    all_processed = False
                    break

            if not all_processed:
                tasks.append((lang, instruction, source_text, resp_clients))

    total_count = len(tasks)
    print(f"\n待处理: {total_count} 条指令")

    if total_count == 0:
        print("\n所有指令已处理完毕！")
        for f in csv_files.values():
            f.close()
        return

    # 并发处理
    processed_count = 0
    print(f"\n开始并发处理 (max_workers={max_workers})...")

    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_item = {executor.submit(process_single_item, item): item for item in tasks}

            for future in as_completed(future_to_item):
                item = future_to_item[future]
                lang, instruction, source_text, _ = item

                try:
                    results = future.result()
                    processed_count += 1

                    print(f"[{processed_count}/{total_count}] {lang} - {source_text[:50]}...")

                    # 写入结果
                    for model_id, row_data in results.items():
                        key = (lang, source_text)
                        if key not in processed_sets[model_id]:
                            with file_locks[model_id]:
                                csv_writers[model_id].writerow(row_data)
                                csv_files[model_id].flush()
                            processed_sets[model_id].add(key)

                except Exception as e:
                    print(f"Error processing {lang} - {source_text[:30]}: {e}")

    finally:
        # Close all CSV files
        for f in csv_files.values():
            f.close()

    print(f"\nStage 1 complete! Outputs saved to {llm_output_base}")
    print("=" * 80)


def run_stage2():
    """
    Stage 2: 将 LLM 输出翻译成英文，写入 CSV

    读取 Stage 1 的 CSV，添加 resp_translated 列
    CSV 结构: source_lang_code, input_text, source_text, model_resp, resp_translated
    """
    print("=" * 80)
    print("STAGE 2: 翻译 LLM 输出为英文")
    print("=" * 80)

    # Load translator config
    translator_config = load_translator_config()
    model_cfg = translator_config.get('model', {})

    # Initialize translator
    print("\nInitializing NLLB translator...")
    translator = NLLBTranslator(
        model_dir=model_cfg.get('model_dir', '~/data/models/nllb-200'),
        model_name=model_cfg.get('model_name', 'facebook/nllb-200-distilled-600M'),
        device=model_cfg.get('device', 'cuda'),
        use_cache=model_cfg.get('use_cache', True)
    )

    llm_output_base = PROJECT_ROOT / 'outputs' / 'llm'

    # Find all stage1 CSV files
    for model_dir in llm_output_base.iterdir():
        if not model_dir.is_dir():
            continue

        model_id = model_dir.name
        stage1_csv = model_dir / f'advbench_output_{model_id}_stage1.csv'
        stage2_csv = model_dir / f'advbench_output_{model_id}_stage2.csv'

        if not stage1_csv.exists():
            print(f"Skipping {model_id}: stage1 CSV not found")
            continue

        print(f"\nProcessing {model_id}...")

        # Read stage1 CSV
        df = pd.read_csv(stage1_csv)
        print(f"  Read {len(df)} rows from stage1")

        # Add resp_translated column
        target_lang_code = 'en'
        resp_translated_list = []

        for idx, row in df.iterrows():
            if idx % 20 == 0:
                print(f"  Translating [{idx+1}/{len(df)}]...")

            model_resp = row['model_resp']
            source_lang_code = row['source_lang_code']

            resp_translated = translate_llm_output(
                translator, model_resp, source_lang_code, target_lang_code
            )
            resp_translated_list.append(resp_translated)

        df['resp_translated'] = resp_translated_list

        # Write stage2 CSV
        df.to_csv(stage2_csv, index=False, encoding='utf-8')
        print(f"  Stage 2 CSV written to {stage2_csv}")

    print(f"\nStage 2 complete!")
    print("=" * 80)

def check_eval_cfg(eval_config) -> bool:
    eval_model_cfg = eval_config.get('model', {})
    if not eval_model_cfg.get('api_key'):
        return False 
    if not eval_model_cfg.get('model_id'):
        return False 
    
def run_stage3():
    """
    Stage 3: 将翻译好的输出输入到评估模型，写入评估结果

    读取 Stage 2 的 CSV，添加 evaluation 列
    CSV 结构: source_lang_code, input_text, source_text, model_resp, resp_translated, evaluation
    """
    print("=" * 80)
    print("STAGE 3: 评估 LLM 响应")
    print("=" * 80)

    # Load evaluation config
    eval_config = load_eval_config()
    if check_eval_cfg(eval_config) is False:
        print("Eval_model config is wrong, skip stage 3")
        return 
    
    eval_model_cfg = eval_config.get('model', {})
    eval_client = create_llm_client(eval_model_cfg)

    llm_output_base = PROJECT_ROOT / 'outputs' / 'llm'

    # Find all stage2 CSV files
    for model_dir in llm_output_base.iterdir():
        if not model_dir.is_dir():
            continue

        model_id = model_dir.name
        stage2_csv = model_dir / f'advbench_output_{model_id}_stage2.csv'
        final_csv = model_dir / f'advbench_output_{model_id}.csv'

        if not stage2_csv.exists():
            print(f"Skipping {model_id}: stage2 CSV not found")
            continue

        print(f"\nProcessing {model_id}...")

        # Read stage2 CSV
        df = pd.read_csv(stage2_csv)
        print(f"  Read {len(df)} rows from stage2")

        # Add evaluation column
        evaluation_list = []

        for idx, row in df.iterrows():
            if idx % 20 == 0:
                print(f"  Evaluating [{idx+1}/{len(df)}]...")

            source_text = row['source_text']
            resp_translated = row['resp_translated']
            model_resp = row['model_resp']

            # Use translated response for evaluation if available
            eval_response_text = resp_translated if resp_translated and not str(resp_translated).startswith('[') else model_resp

            try:
                evaluation = evaluate_response(eval_client, eval_model_cfg, source_text, eval_response_text)
            except Exception as e:
                print(f"  Evaluation error: {e}")
                evaluation = -1

            evaluation_list.append(evaluation)

        df['evaluation'] = evaluation_list

        # Reorder columns to match original format
        final_columns = ['source_lang_code', 'input_text', 'model_resp', 'resp_translated', 'evaluation']
        df = df[final_columns]

        # Write final CSV
        df.to_csv(final_csv, index=False, encoding='utf-8')
        print(f"  Final CSV written to {final_csv}")

    print(f"\nStage 3 complete!")
    print("=" * 80)


def calculate_asr_from_csv(csv_path):
    """
    从 CSV 文件读取并统计攻击成功率 ASR，按语言分类

    Args:
        csv_path: CSV 文件路径

    Returns:
        dict: 格式为 {lang: {'success': int, 'total': int, 'asr': float}}
    """
    df = pd.read_csv(csv_path)
    lang_results = {}

    for lang in df['source_lang_code'].unique():
        lang_df = df[df['source_lang_code'] == lang]
        success_count = lang_df['evaluation'].sum()
        total_count = len(lang_df)
        asr = success_count / total_count if total_count > 0 else 0.0
        lang_results[lang] = {
            'success': int(success_count),
            'total': total_count,
            'asr': asr
        }

    return lang_results


def calculate_asr_from_directory(output_dir):
    """
    从输出目录读取所有模型的 CSV 文件并统计 ASR

    Args:
        output_dir: outputs/llm 目录路径

    Returns:
        dict: 格式为 {model_id: {lang: {'success': int, 'total': int, 'asr': float}}}
    """
    output_dir = Path(output_dir)
    asr_results = {}

    for model_dir in output_dir.iterdir():
        if model_dir.is_dir():
            model_id = model_dir.name
            csv_file = model_dir / f'advbench_output_{model_id}.csv'
            if csv_file.exists():
                asr_results[model_id] = calculate_asr_from_csv(csv_file)

    return asr_results


def print_asr_report(asr_results):
    """
    打印 ASR 统计结果

    Args:
        asr_results: calculate_asr_from_directory 返回的 dict
    """
    print("\n" + "=" * 80)
    print("攻击成功率 (ASR) 统计报告")
    print("=" * 80)

    for model_id, lang_results in asr_results.items():
        print(f"\n【模型: {model_id}】")
        print("-" * 60)

        # Calculate overall ASR for this model
        total_success = sum(r['success'] for r in lang_results.values())
        total_total = sum(r['total'] for r in lang_results.values())
        overall_asr = total_success / total_total if total_total > 0 else 0.0

        # Print per-language results
        for lang, stats in sorted(lang_results.items()):
            print(f"  {lang:10s} | ASR: {stats['asr']:6.2%} | 成功: {stats['success']:3d}/{stats['total']:<3d}")

        print("-" * 60)
        print(f"  {'总体':10s} | ASR: {overall_asr:6.2%} | 成功: {total_success:3d}/{total_total:<3d}")

    print("\n" + "=" * 80)


def run_statistics_only():
    """单独运行统计函数"""
    llm_output_base = PROJECT_ROOT / 'outputs' / 'llm'
    if not llm_output_base.exists():
        print(f"Error: Output directory {llm_output_base} does not exist!")
        return

    asr_results = calculate_asr_from_directory(llm_output_base)
    if asr_results:
        print_asr_report(asr_results)
    else:
        print("No CSV files found for statistics.")


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Extract AdvBench outputs from LLMs')
    parser.add_argument('--stage', '-S', type=int, choices=[1, 2, 3],
                        help='Run specific stage (1, 2, or 3)')
    parser.add_argument('--limit', '-n', type=int, default=None,
                        help='Number of instructions to test per language (Stage 1 only)')
    parser.add_argument('--stats-only', '-s', action='store_true',
                        help='Only run statistics on existing CSV files')
    parser.add_argument('--all', '-a', action='store_true',
                        help='Run all stages (1 -> 2 -> 3)')
    parser.add_argument('--workers', '-w', type=int, default=5,
                        help='Number of concurrent workers for Stage 1 (default: 5)')
    args = parser.parse_args()

    if args.stats_only:
        run_statistics_only()
    elif args.stage == 1:
        run_stage1(limit=args.limit, max_workers=args.workers)
    elif args.stage == 2:
        run_stage2()
    elif args.stage == 3:
        run_stage3()
        # After stage 3, run statistics
        llm_output_base = PROJECT_ROOT / 'outputs' / 'llm'
        asr_results = calculate_asr_from_directory(llm_output_base)
        if asr_results:
            print_asr_report(asr_results)
    elif args.all:
        run_stage1(limit=args.limit, max_workers=args.workers)
        run_stage2()
        run_stage3()
        # After all stages, run statistics
        llm_output_base = PROJECT_ROOT / 'outputs' / 'llm'
        asr_results = calculate_asr_from_directory(llm_output_base)
        if asr_results:
            print_asr_report(asr_results)
    else:
        print("Please specify either --stage N, --all, or --stats-only")
        print("\nUsage:")
        print("  --stage 1 [--limit N] [--workers W]  Run Stage 1: Query LLMs")
        print("  --stage 2                              Run Stage 2: Translate outputs")
        print("  --stage 3                              Run Stage 3: Evaluate responses")
        print("  --all [--limit N] [--workers W]       Run all stages (1->2->3)")
        print("  --stats-only                           Only show statistics from final CSVs")


if __name__ == '__main__':
    main()
