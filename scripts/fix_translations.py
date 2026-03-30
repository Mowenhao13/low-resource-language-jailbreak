import csv
import time
import sys
sys.path.insert(0, '/Users/halllo/projects/local/low-resource-lang-jailbreak/scripts')
from translate.translate import Translator

input_csv = '/Users/halllo/projects/local/low-resource-lang-jailbreak/data/processed/advbench_input.csv'
output_csv = '/Users/halllo/projects/local/low-resource-lang-jailbreak/data/processed/advbench_input_fixed.csv'

translator = Translator()

# 需要修复的行范围（0-based index，不包含表头）
# 810-830 和 1190-1200 是用户说的行号（看起来是从1开始的，包含表头）
# 实际数据行需要减去1（因为表头是第0行）

# 读取所有行
rows = []
with open(input_csv, 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    for row in reader:
        rows.append(row)

header = rows[0]
data_rows = rows[1:]

print(f"总数据行数: {len(data_rows)}")

# 需要检查/修复的行（用户说的行号是从1开始的，包含表头）
# 810-830 对应的 data_rows index: 809-829
# 1190-1200 对应的 data_rows index: 1189-1199

fix_ranges = [
    (809, 830),   # 用户说的 810-830
    (1189, 1200)  # 用户说的 1190-1200
]

fixed_count = 0
error_count = 0

for start_idx, end_idx in fix_ranges:
    for i in range(start_idx, min(end_idx, len(data_rows))):
        row = data_rows[i]
        if len(row) < 3:
            continue

        target_lang = row[0]
        source_text = row[1]
        translated_text = row[2]

        # 检查是否需要修复
        need_fix = False
        if 'ERROR:' in translated_text or 'HTTPSConnectionPool' in translated_text:
            need_fix = True
            print(f"行 {i+2} (用户行号 {i+2}): 发现错误 - {target_lang}")
        # 也可以检查白苗语中是否保留了太多英文词
        elif target_lang == 'hmn':
            english_words = ['cyber attack', 'computer', 'credit card']
            for word in english_words:
                if word in translated_text:
                    need_fix = True
                    print(f"行 {i+2} (用户行号 {i+2}): 发现英文词 '{word}' - {target_lang}")
                    break

        if need_fix:
            try:
                print(f"  重新翻译: {source_text[:60]}... -> {target_lang}")
                new_translation = translator.Translate(source_text, 'en', target_lang)
                data_rows[i][2] = new_translation
                print(f"  结果: {new_translation[:60]}...")
                fixed_count += 1
                time.sleep(0.5)  # 避免限流
            except Exception as e:
                print(f"  翻译失败: {e}")
                error_count += 1

# 写入修复后的文件
with open(output_csv, 'w', encoding='utf-8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(data_rows)

print(f"\n修复完成！")
print(f"修复条目: {fixed_count}")
print(f"失败条目: {error_count}")
print(f"输出文件: {output_csv}")
print("\n请检查后用以下命令替换原文件:")
print(f"mv {output_csv} {input_csv}")
