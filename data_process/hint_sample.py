import json
import os


FILE_CONFIG = {
    "data_file": "/data/qq/LLaMA-Factory/data/Qwen2.5-Math-7B/math-merge/tem_0.7/stage_3-1.jsonl",
    "eval_result_files": [
        "/data/qq/models/Qwen/Qwen2.5-Math-7B/math_eval/math-merge/Recording/qwen_math_7b_level_3_step_1-4_qwen_box_seed0_t0.7_n_sample_16.jsonl",
        "/data/qq/models/Qwen/Qwen2.5-Math-7B/math_eval/math-merge/Recording/qwen_math_7b_level_3_step_1-2_qwen_box_seed0_t0.7_n_sample_16.jsonl",
        "/data/qq/models/Qwen/Qwen2.5-Math-7B/math_eval/math-merge/Recording/qwen_math_7b_level_3_step_3-4_qwen_box_seed0_t0.7_n_sample_16.jsonl"
    ],
    "eval_source_files": [
        "/data/qq/math-evaluation-harness/data/math-merge/qwen_math_7b_level_3_step_1-4.jsonl",
        "/data/qq/math-evaluation-harness/data/math-merge/qwen_math_7b_level_3_step_1-2.jsonl",
        "/data/qq/math-evaluation-harness/data/math-merge/qwen_math_7b_level_3_step_3-4.jsonl"
    ],
    "output_file": "/data/qq/LLaMA-Factory/data/Qwen2.5-Math-7B/math-merge/tem_0.7/stage_3-3.jsonl"
}
# 配置文件路径
# FILE_CONFIG = {
#     "data_file": "/data/qq/LLaMA-Factory/data/Qwen2.5-Math-1.5B/math-merge/tem_0.7/stage_3-1.jsonl",
#     "eval_result_files": [
#         "/data/qq/models/Qwen/Qwen2.5-Math-1.5B/math_eval/math-merge/Recording/qwen_math_1.5b_level_3_step_1-4_qwen_box_seed0_t0.7_n_sample_16.jsonl",
#         "/data/qq/models/Qwen/Qwen2.5-Math-1.5B/math_eval/math-merge/Recording/qwen_math_1.5b_level_3_step_1-2_qwen_box_seed0_t0.7_n_sample_16.jsonl",
#         "/data/qq/models/Qwen/Qwen2.5-Math-1.5B/math_eval/math-merge/Recording/qwen_math_1.5b_level_3_step_3-4_qwen_box_seed0_t0.7_n_sample_16.jsonl"
#     ],
#     "eval_source_files": [
#         "/data/qq/math-evaluation-harness/data/math-merge/qwen_math_1.5b_level_3_step_1-4.jsonl",
#         "/data/qq/math-evaluation-harness/data/math-merge/qwen_math_1.5b_level_3_step_1-2.jsonl",
#         "/data/qq/math-evaluation-harness/data/math-merge/qwen_math_1.5b_level_3_step_3-4.jsonl"
#     ],
#     "output_file": "/data/qq/LLaMA-Factory/data/Qwen2.5-Math-1.5B/math-merge/tem_0.7/stage_3-3.jsonl"
# }

# ---------------------- 辅助函数 ----------------------
def validate_paths():
    """验证所有输入文件是否存在"""
    required_files = [FILE_CONFIG["data_file"]] + FILE_CONFIG["eval_result_files"] + FILE_CONFIG["eval_source_files"]
    for path in required_files:
        if not os.path.exists(path):
            raise FileNotFoundError(f"关键文件缺失: {path}")

def load_jsonl_safe(path: str) -> list:
    """安全加载JSONL文件，带异常处理"""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return [json.loads(line) for line in f]
    except Exception as e:
        raise RuntimeError(f"加载文件失败 {path}: {str(e)}")

def build_eval_maps():
    """构建评测结果和源数据的快速查找字典"""
    eval_scores = []
    for result_file in FILE_CONFIG["eval_result_files"]:
        score_map = {}
        for item in load_jsonl_safe(result_file):
            # 直接检查True是否存在于score列表
            score_list = item.get('score', [])
            if not isinstance(score_list, list):
                print(f"警告：idx {item['idx']} 的score字段类型异常 ({type(score_list)})，已跳过")
                continue
            score_map[item['idx']] = True in score_list
        eval_scores.append(score_map)
    
    eval_sources = []
    for source_file in FILE_CONFIG["eval_source_files"]:
        eval_sources.append({item['idx']: item for item in load_jsonl_safe(source_file)})
    
    return eval_scores, eval_sources

# ---------------------- 主逻辑 ----------------------
def main():
    validate_paths()
    
    # 加载数据
    main_data = load_jsonl_safe(FILE_CONFIG["data_file"])
    eval_scores, eval_sources = build_eval_maps()
    
    # 处理数据
    stats = {'replaced': [0, 0, 0], 'discarded': 0}
    updated_data = []
    
    for item in main_data:
        # 跳过不需要修复的数据
        if item.get('accuracy', 0) != 0:
            updated_data.append(item)
            continue
        
        idx = item.get('idx')
        if idx is None:
            print(f"警告：发现无idx字段的数据项，已丢弃。内容: {item}")
            stats['discarded'] += 1
            continue
        
        replaced = False
        for eval_idx in range(3):  # 按优先级eval1 → eval3处理
            # 检查评测是否通过
            if not eval_scores[eval_idx].get(idx, False):
                continue
            
            # 获取源数据
            source = eval_sources[eval_idx].get(idx)
            if not source:
                print(f"警告：eval{eval_idx+1} idx {idx} 评测通过但源数据缺失")
                continue
            
            # 执行替换
            try:
                new_item = item.copy()
                new_item['question'] = source['problem']
                new_item['answer'] = source['answer_end']
                updated_data.append(new_item)
                stats['replaced'][eval_idx] += 1
                replaced = True
                break  # 找到有效替换即终止
            except KeyError as e:
                print(f"错误：eval{eval_idx+1} idx {idx} 源数据缺失字段 {e}")
        
        if not replaced:
            stats['discarded'] += 1
    
    # 保存结果
    with open(FILE_CONFIG["output_file"], 'w', encoding='utf-8') as f:
        for item in updated_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    # 打印报告
    print("\n✅ 处理完成！统计如下：")
    print(f"  从 eval1 替换: {stats['replaced'][0]}")
    print(f"  从 eval2 替换: {stats['replaced'][1]}")
    print(f"  从 eval3 替换: {stats['replaced'][2]}")
    print(f"❌ 丢弃总数: {stats['discarded']}")
    print(f"📦 最终数据量: {len(updated_data)}/{len(main_data)}")

if __name__ == "__main__":
    main()