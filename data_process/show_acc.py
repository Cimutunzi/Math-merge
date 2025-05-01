import json

# 评测结果文件
eval_result_files=[
        "/data/qq/models/Qwen/Qwen2.5-Math-7B/math_eval/math-merge/Recording/qwen_math_7b_level_3_step_1-4_qwen_box_seed0_t0.0_n_sample_16.jsonl",
        "/data/qq/models/Qwen/Qwen2.5-Math-7B/math_eval/math-merge/Recording/qwen_math_7b_level_3_step_1-2_qwen_box_seed0_t0.0_n_sample_16.jsonl",
        "/data/qq/models/Qwen/Qwen2.5-Math-7B/math_eval/math-merge/Recording/qwen_math_7b_level_3_step_3-4_qwen_box_seed0_t0.0_n_sample_16.jsonl"
    ]

for i, path in enumerate(eval_result_files):
    zero_acc_count = 0
    total = 0
    a = 0
    b = 0
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            score_list = item.get('score', []) 
            total += 1
            if isinstance(score_list, list):
                if sum(bool(x) for x in score_list) == 0:
                    zero_acc_count += 1
                    a += 1
                    if a%100 == 0:
                        print(score_list)
            else:
                if not score_list:
                    zero_acc_count += 1
                    b += 1
    print(f"📊 eval{i+1}: 总数 {total}，acc==0 的样本数：{zero_acc_count}")
