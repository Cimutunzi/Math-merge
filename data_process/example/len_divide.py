import json

def split_by_dot_with_ratio(text: str, ratio: float = 0.5):
    target_index = int(len(text) * ratio)
    dot_positions = [i for i, c in enumerate(text) if c == '  ']

    if not dot_positions:
        return text[:target_index].strip(), text[target_index:].strip()

    closest_dot = min(dot_positions, key=lambda x: abs(x - target_index))
    split_index = closest_dot + 1  # 保留句号
    return text[:split_index].strip(), text[split_index:].strip()

def process_jsonl_file_fixed_fields(input_path, output_path, ratio=0.5):
    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:

        for line in infile:
            try:
                item = json.loads(line)
                idx = item.get('idx')
                question = item.get('question', '').strip()
                answer = item.get('solution', '').strip()

                if not answer:
                    continue

                sol_start, sol_end = split_by_dot_with_ratio(answer, ratio)

                output_item = {
                    'idx': idx,
                    'problem': f"{question}\n\n{sol_start}",
                    'solution': answer,
                    'answer_end': sol_end
                }

                outfile.write(json.dumps(output_item, ensure_ascii=False) + '\n')

            except json.JSONDecodeError:
                print("跳过无法解析的行：", line[:100])


# 使用方法
process_jsonl_file_fixed_fields(
    '/data/qq/models/Qwen/Qwen2.5-Math-1.5B/math_eval/math-merge/data/tem_0.7/level_3.jsonl', 
    '/data/qq/math-evaluation-harness/data/math-merge/qwen_math_1.5b_level_3_step_3-4.jsonl', 
    ratio=0.75
    )
