import json
import jsonlines
import matplotlib.pyplot as plt
from collections import defaultdict

# ================== 配置参数 ==================
STAGE = 'single_3'
LEVEL_FILES = {
    "level 1": "/data/qq/models/Qwen/Qwen2.5-Math-7B/math_eval/math-merge/data/tem_0.7/level_1.jsonl",
    "level 2": "/data/qq/models/Qwen/Qwen2.5-Math-7B/math_eval/math-merge/data/tem_0.7/level_2.jsonl", 
    "level 3": "/data/qq/models/Qwen/Qwen2.5-Math-7B/math_eval/math-merge/data/tem_0.7/level_3.jsonl"
}
RESULT_FILE = "/data/qq/models/Qwen/Qwen2.5-Math-7B/math_eval/math-merge/Stage_3_tem_0.7/single/5.0e-05/math-merge/Recording/train_qwen_box_seed0_t0.0_n_sample_1.jsonl"
OUTPUT_DIR = "/data/qq/models/Qwen/Qwen2.5-Math-7B/math_eval/math-merge/level_performance/"

# ================== 数据准备 ==================
def build_id_mapping():
    """建立ID到真实level的映射"""
    id_to_level = {}
    
    for level_name, file_path in LEVEL_FILES.items():
        try:
            with jsonlines.open(file_path) as reader:
                for obj in reader:
                    if 'idx' in obj:
                        id_to_level[obj['idx']] = level_name
                    else:
                        print(f"Warning: Missing idx in {file_path}")
        except Exception as e:
            print(f"Error reading {file_path}: {str(e)}")
    
    return id_to_level

# ================== 数据分析 ==================
def analyze_results(id_mapping):
    """执行核心分析逻辑"""
    stats = {
        'level_counts': defaultdict(int),
        'correct_counts': defaultdict(int),
        'total_counts': defaultdict(int),
        'missing_ids': set()
    }

    try:
        with jsonlines.open(RESULT_FILE) as reader:
            for obj in reader:
                # 数据完整性检查
                if 'idx' not in obj:
                    print("Warning: Missing idx in result record")
                    continue
                
                idx = obj['idx']
                true_level = id_mapping.get(idx)
                
                # 等级匹配检查
                if not true_level:
                    stats['missing_ids'].add(idx)
                    continue
                
                # 数据有效性检查
                if 'score' not in obj:
                    print(f"Warning: Missing score for idx {idx}")
                    continue
                
                # 统计计算
                scores = obj['score']
                correct = sum(scores)
                total = len(scores)
                
                stats['level_counts'][true_level] += 1
                stats['correct_counts'][true_level] += correct
                stats['total_counts'][true_level] += total
                
    except Exception as e:
        print(f"Error processing result file: {str(e)}")
    
    return stats

# ================== 可视化 ==================
def visualize_results(stats, stage):
    """生成可视化图表"""
    # 准备数据
    levels = ['level 1', 'level 2', 'level 3']
    correct = [stats['correct_counts'][l] for l in levels]
    total = [stats['total_counts'][l] for l in levels]
    accuracy = [c/t if t >0 else 0 for c, t in zip(correct, total)]
    
    # 创建画布
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 柱状图
    bars1 = ax1.bar(levels, total, color='#1f77b4', label='Total Answers')
    bars2 = ax1.bar(levels, correct, color='#ff7f0e', label='Correct Answers')
    ax1.set_title(f'Answer Distribution - Stage {stage}')
    ax1.set_ylabel('Count')
    ax1.legend()
    
    # 添加数值标签
    for bar in bars1 + bars2:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.0f}',
                ha='center', va='bottom')
    
    # 正确率曲线
    ax2.plot(levels, accuracy, marker='o', linestyle='--', color='#2ca02c')
    ax2.set_ylim(0, 1)
    ax2.set_title(f'Accuracy Rate - Stage {stage}')
    ax2.set_ylabel('Accuracy')
    ax2.grid(True)
    
    # 添加数据标签
    for x, y in zip(levels, accuracy):
        ax2.text(x, y, f'{y:.2%}', ha='center', va='bottom')
    
    # 保存结果
    plt.tight_layout()
    output_path = f"{OUTPUT_DIR}level_analysis_{stage}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Results saved to {output_path}")
    
    # 控制台输出
    print("\n=== Analysis Summary ===")
    print(f"{'Level':<10} | {'Problems':<8} | {'Correct':<8} | {'Total':<8} | {'Accuracy':<8}")
    print("-"*45)
    for level in levels:
        print(f"{level:<10} | {stats['level_counts'][level]:<8} | "
              f"{stats['correct_counts'][level]:<8} | {stats['total_counts'][level]:<8} | "
              f"{accuracy[levels.index(level)]:.2%}")
    
    if stats['missing_ids']:
        print(f"\nWarning: Found {len(stats['missing_ids'])} unmatched IDs")

# ================== 主流程 ==================
if __name__ == "__main__":
    print("Starting analysis...")
    
    # 步骤1：建立ID映射
    print("\nBuilding ID mapping...")
    id_mapping = build_id_mapping()
    print(f"Mapped {len(id_mapping)} unique IDs")
    
    # 步骤2：分析结果
    print("\nAnalyzing results...")
    stats = analyze_results(id_mapping)
    
    # 步骤3：可视化展示
    print("\nGenerating visualizations...")
    visualize_results(stats, STAGE)
    
    print("\nAnalysis completed!")