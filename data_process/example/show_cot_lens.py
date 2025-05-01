import json

def compare_solution_and_cot(input_path):
    count = 0

    with open(input_path, 'r', encoding='utf-8') as infile:
        for line in infile:
            try:
                item = json.loads(line)
                solution = item.get('solution', '').strip()
                cot = item.get('cot', '').strip()

                # 比较 solution 和 cot 的长度
                if len(solution) > len(cot):
                    count += 1
            except json.JSONDecodeError:
                print("跳过无法解析的行：", line[:100])

    return count

# 使用方法
input_file = '/data/qq/data_process/example/cot.jsonl'
count = compare_solution_and_cot(input_file)
print(f"Solution 比 Cot 长的数据条数: {count}")
