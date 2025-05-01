import json
import math

input_file = '/data/qq/math-evaluation-harness/data/math-merge/step_merged.jsonl'
output_file = '/data/qq/data_process/example/end_solution_1-2.jsonl'

# 比例参数（你可以随便改，比如 0.5 表示一半，0.75 表示四分之三）
ratio = 0.5

def process_solution(item):
    solution = item.get("solution", "")
    steps = solution.split("<step-devide>")
    step_count = len(steps)

    item["step_devide_count"] = step_count - 1  # 分隔符数

    # 计算拼接到question的步骤数（向下取整）
    suffix_count = math.floor(step_count * ratio)
    # prefix_steps = ''.join(steps[:prefix_count]).strip()
    suffix_steps = ''.join(steps[suffix_count:]).strip()

    # 计算后1/2部分的步骤（截取solution的后半部分）
    # mid = len(steps) // 2  # 中点
    answer_end = suffix_steps

    # original_question = item.get("question", "").strip()
    # item["question"] = prefix_steps + " " + original_question if prefix_steps else original_question
    # item["solution"] = suffix_steps
    item["answer_end"] = answer_end  # 添加answer_end字段

    return item

with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
    for line in infile:
        data = json.loads(line)
        processed = process_solution(data)
        outfile.write(json.dumps(processed, ensure_ascii=False) + '\n')
