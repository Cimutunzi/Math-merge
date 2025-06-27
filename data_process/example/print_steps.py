import re
import json

def extract_math_blocks(content):
    pattern = r'\\begin{(align\*?|aligned|gather\*?|enumerate|itemize)}.*?\\end{\1}'
    blocks = []
    matches = []

    def replace_block(match):
        blocks.append(match.group(0))
        matches.append({
            "start": match.start(),
            "end": match.end(),
            "block": match.group(0)
        })
        return f"__MATH_BLOCK_{len(blocks) - 1}__"

    text = re.sub(pattern, replace_block, content, flags=re.DOTALL)
    return text, blocks, matches

def restore_math_blocks(text_parts, blocks):
    restored = []
    for part in text_parts:
        for i, block in enumerate(blocks):
            part = part.replace(f"__MATH_BLOCK_{i}__", block)
        restored.append(part.strip())
    return [p for p in restored if p]

def split_natural_sentences(content):
    parts = re.split(r'(?<=[。！？!?\.])\s+|\n\s*\n+', content)
    return [p.strip() for p in parts if p.strip()]

def process_latex_steps_with_math_matches(content):
    text_with_placeholders, blocks, matches = extract_math_blocks(content)
    text_parts = split_natural_sentences(text_with_placeholders)
    restored_steps = restore_math_blocks(text_parts, blocks)
    return {
        "steps": restored_steps,
        "math_blocks": matches
    }

def analyze_solution_by_idx(file_path, target_idx):
    with open(file_path, 'r', encoding='utf-8') as infile:
        for line in infile:
            item = json.loads(line)
            if str(item.get("idx")) == str(target_idx):
                print(f"[INFO] Found idx: {target_idx}")
                solution = item.get("solution", "")
                result = process_latex_steps_with_math_matches(solution)

                print("\n== Steps ==")
                for i, step in enumerate(result["steps"]):
                    # print(f"Step {i+1}:\n{step}\n{'-'*40}")
                    print(f"Step {i+1}:\n{step}\n")

                print("\n== Math Blocks ==")
                for i, block_info in enumerate(result["math_blocks"]):
                    print(f"Block {i+1}:")
                    print(f"Start: {block_info['start']}, End: {block_info['end']}")
                    print(f"Content:\n{block_info['block']}\n{'='*40}")
                return

        print(f"[WARN] idx: {target_idx} not found.")


if __name__ == "__main__":
    jsonl_path = '/data/qq/math-evaluation-harness/data/math-merge/train.jsonl'
    target_idx = 4155  # 替换为你想查找的具体idx
    analyze_solution_by_idx(jsonl_path, target_idx)
