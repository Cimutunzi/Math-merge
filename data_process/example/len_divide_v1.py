import re
import math
import json

def extract_math_blocks(content):
    pattern = r'\\begin{(align\*?|aligned|gather\*?|enumerate|itemize)}.*?\\end{\1}'
    blocks = []
    def replace_block(match):
        blocks.append(match.group(0))
        return f"__MATH_BLOCK_{len(blocks) - 1}__"
    text = re.sub(pattern, replace_block, content, flags=re.DOTALL)
    return text, blocks

def restore_math_blocks(text_parts, blocks):
    restored = []
    for part in text_parts:
        for i, block in enumerate(blocks):
            part = part.replace(f"__MATH_BLOCK_{i}__", block)
        restored.append(part.strip())
    return [p for p in restored if p]

def split_natural_sentences(content):
    # 基于句号、换行、空行来切分自然语言
    parts = re.split(r'(?<=[。！？!?\.])\s+|\n\s*\n+', content)
    return [p.strip() for p in parts if p.strip()]

def process_latex_steps(content):
    text_with_placeholders, blocks = extract_math_blocks(content)
    text_parts = split_natural_sentences(text_with_placeholders)
    steps = restore_math_blocks(text_parts, blocks)
    return steps

def split_steps_with_ratio(steps, ratio=0.75):
    if not steps:
        return [], []

    split_index = max(1, math.ceil(len(steps) * ratio))
    split_index = min(split_index, len(steps) - 1)

    question_steps = steps[:split_index]
    answer_end_steps = steps[split_index:]

    if not question_steps:
        question_steps = [steps[0]]
        answer_end_steps = steps[1:]

    return question_steps, answer_end_steps

def process_dataset_from_file(input_path, output_path, ratio=0.75):
    kept = 0
    skipped = 0

    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:
        for line in infile:
            item = json.loads(line)
            question = item.get("question", "")
            # answer = item.get("answer", "")
            # answer = item.get("solution", "")

            # question = item.get("problem", "")
            answer = item.get("solution", "")

            steps = process_latex_steps(answer)
            if len(steps) < 2:
                skipped += 1
                continue

            question_steps, answer_end_steps = split_steps_with_ratio(steps, ratio)
           
            q_len = len("".join(question_steps))
            a_len = len("".join(answer_end_steps))
            if 1.5*q_len >= a_len:
                skipped += 1
                continue


            if not answer_end_steps:
                skipped += 1
                continue

            new_item = {
                "idx": item.get("idx"),
                "question": f"{question}\n\nBelow is a partial solution. Please continue solving from where it left off:\n" + "\n".join(question_steps),
                "solution": answer,
                "answer_end": "\n".join(answer_end_steps),
                "level": item.get("level"),
                "accuracy": item.get("accuracy")
            }

            outfile.write(json.dumps(new_item, ensure_ascii=False) + '\n')
            kept += 1

    print(f"[INFO] Finished: kept {kept}, skipped {skipped}")

# 示例调用
if __name__ == "__main__":
    process_dataset_from_file(
        '/data/qq/models/deepseek-math-7b-instruct/math_eval/math-merge/data/tem_0.7/level_3.jsonl', 
        '/data/qq/math-evaluation-harness/data/math-merge/deepseek-math-7b-instruct_level_3_hint.jsonl', 
        ratio=0.25
    )
    # process_dataset_from_file(
    #     '/data/qq/models/Qwen/Qwen2.5-Math-1.5B/math_eval/math-merge/data/tem_0.7/level_3.jsonl', 
    #     '/data/qq/data/math-merge/data/Qwen2.5-Math-1.5B/qwen_math_1.5b_level_3_step_1-4_v1.jsonl', 
    #     ratio=0.25
    # )