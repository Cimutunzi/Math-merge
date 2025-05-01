import json
import math

input_file = '/data/qq/math-evaluation-harness/data/math-merge/step_merged.jsonl'
output_file = '/data/qq/math-evaluation-harness/data/math-merge/step_3-4_revise.jsonl'

ratio = 0.75


def process_solution(item):
    solution = item.get("solution", "").replace("\b", "\\b").replace("\r", "\\r").replace("\f", "\\f")
    steps = solution.split("<step-devide>")
    step_count = len(steps)

    item["step_devide_count"] = step_count - 1  # 分隔符数

    # 计算拼接到question的步骤数（向下取整）
    prefix_count = math.floor(step_count * ratio)
    prefix_steps = ''.join(steps[:prefix_count]).strip()
    suffix_steps = ''.join(steps[prefix_count:]).strip()

    original_question = item.get("problem", "").strip()
    item["problem"] = original_question + " " + prefix_steps if prefix_steps else original_question
    item["solution"] = suffix_steps

    return item

with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
    for line in infile:
        data = json.loads(line)
        processed = process_solution(data)
        outfile.write(json.dumps(processed, ensure_ascii=False) + '\n')
