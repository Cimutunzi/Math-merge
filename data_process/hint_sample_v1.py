import json
import os


# FILE_CONFIG = {
#     "data_file": "/data/qq/LLaMA-Factory/data/Qwen2.5-Math-7B/math-merge/tem_0.7/stage_3-1.jsonl",
#     "eval_result_files": [
#         "/data/qq/models/Qwen/Qwen2.5-Math-7B/math_eval/math-merge/Recording/qwen_math_7b_level_3_step_1-4_qwen_box_seed0_t0.7_n_sample_16.jsonl",
#         "/data/qq/models/Qwen/Qwen2.5-Math-7B/math_eval/math-merge/Recording/qwen_math_7b_level_3_step_1-2_qwen_box_seed0_t0.7_n_sample_16.jsonl",
#         "/data/qq/models/Qwen/Qwen2.5-Math-7B/math_eval/math-merge/Recording/qwen_math_7b_level_3_step_3-4_qwen_box_seed0_t0.7_n_sample_16.jsonl"
#     ],
#     "eval_source_files": [
#         "/data/qq/math-evaluation-harness/data/math-merge/qwen_math_7b_level_3_step_1-4.jsonl",
#         "/data/qq/math-evaluation-harness/data/math-merge/qwen_math_7b_level_3_step_1-2.jsonl",
#         "/data/qq/math-evaluation-harness/data/math-merge/qwen_math_7b_level_3_step_3-4.jsonl"
#     ],
#     "output_file": "/data/qq/LLaMA-Factory/data/Qwen2.5-Math-7B/math-merge/tem_0.7/stage_3-4.jsonl"
# }

配置文件路径
FILE_CONFIG = {
    "data_file": "/data/qq/LLaMA-Factory/data/Qwen2.5-Math-1.5B/math-merge/tem_0.7/stage_3-1.jsonl",
    "eval_result_files": [
        "/data/qq/models/Qwen/Qwen2.5-Math-1.5B/math_eval/math-merge/Recording/qwen_math_1.5b_level_3_step_1-4_qwen_box_seed0_t0.7_n_sample_16.jsonl",
        "/data/qq/models/Qwen/Qwen2.5-Math-1.5B/math_eval/math-merge/Recording/qwen_math_1.5b_level_3_step_1-2_qwen_box_seed0_t0.7_n_sample_16.jsonl",
        "/data/qq/models/Qwen/Qwen2.5-Math-1.5B/math_eval/math-merge/Recording/qwen_math_1.5b_level_3_step_3-4_qwen_box_seed0_t0.7_n_sample_16.jsonl"
    ],
    "eval_source_files": [
        "/data/qq/math-evaluation-harness/data/math-merge/qwen_math_1.5b_level_3_step_1-4.jsonl",
        "/data/qq/math-evaluation-harness/data/math-merge/qwen_math_1.5b_level_3_step_1-2.jsonl",
        "/data/qq/math-evaluation-harness/data/math-merge/qwen_math_1.5b_level_3_step_3-4.jsonl"
    ],
    "output_file": "/data/qq/LLaMA-Factory/data/Qwen2.5-Math-1.5B/math-merge/tem_0.7/stage_3-4.jsonl"
}

def validate_paths():
    required_files = [FILE_CONFIG["data_file"]] + FILE_CONFIG["eval_result_files"] + FILE_CONFIG["eval_source_files"]
    for path in required_files:
        if not os.path.exists(path):
            raise FileNotFoundError(f"缺失文件: {path}")

def load_jsonl_safe(path: str) -> list:
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

def build_eval_maps():
    eval_scores = []
    for result_file in FILE_CONFIG["eval_result_files"]:
        score_map = {}
        for item in load_jsonl_safe(result_file):
            score_list = item.get('score', [])
            if isinstance(score_list, list):
                score_map[item['idx']] = score_list
        eval_scores.append(score_map)

    eval_sources = []
    for source_file in FILE_CONFIG["eval_source_files"]:
        source_map = {item['idx']: item for item in load_jsonl_safe(source_file)}
        eval_sources.append(source_map)

    return eval_scores, eval_sources

def main():
    validate_paths()

    main_data = load_jsonl_safe(FILE_CONFIG["data_file"])
    eval_scores, eval_sources = build_eval_maps()

    stats = {'replaced': [0, 0, 0], 'discarded': 0}
    updated_data = []

    for item in main_data:
        idx = item.get('idx')
        if idx is None:
            print(f"⚠️ 无 idx，跳过：{item}")
            stats['discarded'] += 1
            continue

        best_eval_idx = -1
        best_score_ratio = -1.0

        for eval_idx in range(3):
            score_list = eval_scores[eval_idx].get(idx)
            if isinstance(score_list, list) and len(score_list) > 0:
                acc = sum(bool(x) for x in score_list) / len(score_list)
                if acc > best_score_ratio:
                    best_score_ratio = acc
                    best_eval_idx = eval_idx

        if best_eval_idx == -1:
            # idx 不在任何评测中，保留原样
            updated_data.append(item)
            continue

        if best_score_ratio < 0.25:
            # 虽然存在，但准确率过低，丢弃
            stats['discarded'] += 1
            continue

        # 替换字段
        source = eval_sources[best_eval_idx].get(idx)
        if not source:
            print(f"⚠️ idx {idx} 在 eval{best_eval_idx+1} 中没有找到源数据")
            stats['discarded'] += 1
            continue

        try:
            new_item = item.copy()
            new_item['question'] = source['problem']
            new_item['answer'] = source['answer_end']
            updated_data.append(new_item)
            stats['replaced'][best_eval_idx] += 1
        except KeyError as e:
            print(f"⚠️ idx {idx} 替换失败，字段缺失: {e}")
            stats['discarded'] += 1

    # 写入新数据
    with open(FILE_CONFIG["output_file"], 'w', encoding='utf-8') as f:
        for item in updated_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    # 打印报告
    print("\n✅ 替换完成，结果如下：")
    for i in range(3):
        print(f"  从 eval{i+1} 替换：{stats['replaced'][i]}")
    print(f"❌ 丢弃数量：{stats['discarded']}")
    print(f"📦 保留数据：{len(updated_data)} / {len(main_data)}")

        # 计算替换后 answer / solution 比例
    answer_ratios = []
    for item in updated_data:
        answer = item.get("answer", "")
        full_solution = item.get("solution", "") or item.get("full_answer", "")
        if full_solution and len(full_solution) > 0:
            answer_ratios.append(len(answer) / len(full_solution))

    if answer_ratios:
        avg_ratio = sum(answer_ratios) / len(answer_ratios)
        print(f"\n📏 替换后 answer_end / solution 的平均长度比值：{avg_ratio:.3f}")
    else:
        print("\n⚠️ 无法计算替换后比值：找不到 solution 字段或无有效数据。")

        print("\n📊 eval_source_files 中 answer_end / solution 比例统计：")
    for i, source_file in enumerate(FILE_CONFIG["eval_source_files"]):
        data = load_jsonl_safe(source_file)
        ratios = []
        for item in data:
            sol = item.get("solution", "")
            ans_end = item.get("answer_end", "")
            if sol and len(sol) > 0:
                ratios.append(len(ans_end) / len(sol))
        if ratios:
            avg = sum(ratios) / len(ratios)
            print(f"  eval{i+1}: {len(ratios)} 条记录，平均比值：{avg:.3f}")
        else:
            print(f"  eval{i+1}: 无可用记录。")

if __name__ == "__main__":
    main()
