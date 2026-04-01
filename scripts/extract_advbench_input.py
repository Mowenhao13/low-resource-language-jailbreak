import pandas as pd
import csv
import sys
import time
import os
from pathlib import Path

# Add scripts directory to path for nllb_translator import
sys.path.insert(0, str(Path(__file__).parent))
from translate.nllb_translator import NLLBTranslator, get_nllb_lang_code

# Get project root directory
PROJECT_ROOT = Path(__file__).parent.parent


def main():
    # 加载数据
    behaviors_df = pd.read_csv(PROJECT_ROOT / 'data' / 'advbench' / 'harmful_behaviors.csv')
    strings_df = pd.read_csv(PROJECT_ROOT / 'data' / 'advbench' / 'harmful_strings.csv')

    print(f"harmful_behaviors: {len(behaviors_df)} 条指令")
    print(f"harmful_strings: {len(strings_df)} 条字符串")

    # 提取有害指令（使用 goal 列）
    all_harmful_instructions = behaviors_df['goal'].tolist()
    # 先测试5条指令，成功后可以增加
    test_count = 100
    harmful_instructions = all_harmful_instructions[:test_count]
    print(f"选择前 {len(harmful_instructions)} 条指令进行翻译 (测试模式)")

    langs = ['zu', 'gd', 'xh', 'gn', 'uk', 'bn', 'th', 'he', 'zh-CN', 'ar', 'it', 'hi']
    print(f"目标语言数量: {len(langs)}")
    print(f"语言列表: {langs}")
    print(f"预计翻译条目: {len(harmful_instructions)} × {len(langs)} = {len(harmful_instructions) * len(langs)}")
    translator = NLLBTranslator(model_dir='~/data/models/nllb-200', device='cuda')

    # CSV输出文件
    output_csv = PROJECT_ROOT / 'data' / 'processed' / 'advbench_input.csv'

    # 写入CSV文件
    with open(output_csv, 'w', encoding='utf-8', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # 写入表头
        writer.writerow(['target_lang_code', 'source_text', 'translated_text'])

        source_lang_code = 'en'
        total_count = len(harmful_instructions) * len(langs)
        processed_count = 0

        for instruction in harmful_instructions:
            source_text = instruction
            for lang in langs:
                target_lang_code = lang
                try:
                    target_text = translator.translate(
                        source_text,
                        source_lang=get_nllb_lang_code(source_lang_code),
                        target_lang=get_nllb_lang_code(target_lang_code)
                    )

                    # 写入CSV行
                    writer.writerow([target_lang_code, source_text, target_text])
                    processed_count += 1

                    # 进度显示
                    if processed_count % 10 == 0:
                        print(f"进度: {processed_count}/{total_count} ({processed_count/total_count*100:.1f}%) - {target_lang_code}")

                    # NLLB不需要API频率限制，不需要sleep

                except Exception as e:
                    print(f"翻译失败: 语言={target_lang_code}, 指令={source_text[:50]}...")
                    print(f"错误详情: {e}")
                    import traceback
                    # 写入错误信息
                    error_msg = f"ERROR: {str(e)[:200]}"
                    writer.writerow([target_lang_code, source_text, error_msg])
                    processed_count += 1
                    continue

    print(f"翻译完成！结果已保存到 {output_csv}")
    print(f"总条目: {processed_count}/{total_count}")


if __name__ == "__main__":
    main()
