import json
import random

def merge_and_shuffle_jsonl(file1, file2, output_file, seed=42):
    all_data = []

    # 读取 file1
    with open(file1, 'r', encoding='utf-8') as f1:
        for line in f1:
            item = json.loads(line)
            all_data.append(item)

    # 读取 file2
    with open(file2, 'r', encoding='utf-8') as f2:
        for line in f2:
            item = json.loads(line)
            all_data.append(item)

    # 打乱顺序
    random.seed(seed)
    random.shuffle(all_data)

    # 写入输出文件
    with open(output_file, 'w', encoding='utf-8') as fout:
        for item in all_data:
            fout.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"[INFO] Merged {len(all_data)} samples from two files into {output_file}")


merge_and_shuffle_jsonl(
    file1="/data/qq/data/math-merge/data/Qwen2.5-Math-7B/stage_3-only_level_2.jsonl",
    file2="/data/qq/data/math-merge/data/Qwen2.5-Math-7B/stage_3-only_25.jsonl",
    output_file="/data/qq/data/math-merge/data/Qwen2.5-Math-7B/stage3-25-level2.jsonl"
)


# merge_and_shuffle_jsonl(
#     file1="/data/qq/data/math-merge/data/Qwen2.5-Math-1.5B/stage_3-only_level_2.jsonl",
#     file2="/data/qq/data/math-merge/data/Qwen2.5-Math-1.5B/stage_3-only_25.jsonl",
#     output_file="/data/qq/data/math-merge/data/Qwen2.5-Math-1.5B/stage3-25-level2.jsonl"
# )