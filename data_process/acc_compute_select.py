import json

def filter_by_score_and_replace_solution(
    eval_file,              # 评测结果文件（包含 score、idx、question、solution）
    original_file,          # 原始数据文件（包含 answer_end）
    output_file,            # 输出保存路径
    score_threshold=0.5     # 准确率阈值（例如 0.5 表示至少答对一半）
):
    # 加载原始数据：建立 idx -> answer_end 映射
    idx_to_answer_end = {}
    with open(original_file, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            idx = item.get("idx")
            answer_end = item.get("answer_end", "")
            if idx is not None:
                idx_to_answer_end[idx] = answer_end

    kept = 0
    total = 0

    with open(eval_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        for line in infile:
            total += 1
            item = json.loads(line)
            score_list = item.get("score", [])
            if not score_list:
                continue

            accuracy = sum(score_list) / len(score_list)

            if accuracy >= score_threshold:
                idx = item.get("idx")
                if idx not in idx_to_answer_end:
                    continue

                # 构造输出条目
                new_item = {
                    "idx": idx,
                    "question": item.get("question", ""),
                    "answer": idx_to_answer_end[idx],  # 用原始数据的 answer_end 替换
                }
                outfile.write(json.dumps(new_item, ensure_ascii=False) + '\n')
                kept += 1

    print(f"[INFO] Finished: kept {kept}/{total} samples with accuracy >= {score_threshold}")

# filter_by_score_and_replace_solution(
#     eval_file="/data/qq/models/Qwen/Qwen2.5-Math-7B/math_eval/math-merge/Recording/qwen_math_7b_level_3_step_1-4_v1_qwen_box_seed0_t0.7_n_sample_16.jsonl",
#     original_file="/data/qq/math-evaluation-harness/data/math-merge/qwen_math_7b_level_3_step_1-4_v1.jsonl",
#     output_file="/data/qq/data/math-merge/data/Qwen2.5-Math-7B/stage_3-only_25.jsonl",
#     score_threshold=0.25  # 只保留准确率 >= 50% 的样本
# )


filter_by_score_and_replace_solution(
    eval_file="/data/qq/models/deepseek-math-7b-instruct/math_eval/math-merge/Recording/deepseek-math-7b-instruct_level_3_hint_deepseek-math_seed0_t0.7_n_sample_16.jsonl",
    original_file="/data/qq/math-evaluation-harness/data/math-merge/deepseek-math-7b-instruct_level_3_hint.jsonl",
    output_file="/data/qq/data/math-merge/data/deepseek-math-7b-instruct/level_3-hint.jsonl",
    score_threshold=0.25  # 只保留准确率 >= 50% 的样本
)