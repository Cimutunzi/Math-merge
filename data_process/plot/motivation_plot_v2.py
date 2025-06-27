import json
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter

def load_level_results(m1_path, m2_path, m3_path):
    """加载三个数据集的等级结果"""
    results = {}
    
    try:
        # 加载模型结果
        m1_results = load_single_model(m1_path)
        m2_results = load_single_model(m2_path)
        m3_results = load_single_model(m3_path)
        
        # 验证一致性并获取共同问题ID
        common_ids = set(m1_results.keys()) & set(m2_results.keys()) & set(m3_results.keys())
        if not common_ids:
            print("无共同问题ID，无法比较")
            return results
        
        # 按等级分组
        level_groups = {}
        for idx in common_ids:
            level = m1_results[idx]['level']
            level_groups.setdefault(level, []).append(idx)
        
        # 统计各等级指标
        for level, ids in level_groups.items():
            total = len(ids)
            m1_acc = sum(np.mean(m1_results[idx]['scores']) for idx in ids)/total
            m2_acc = sum(np.mean(m2_results[idx]['scores']) for idx in ids)/total
            m3_acc = sum(np.mean(m3_results[idx]['scores']) for idx in ids)/total
            results[level] = {
                'total': total,
                'm1_acc': m1_acc,
                'm2_acc': m2_acc,
                'm3_acc': m3_acc,
            }
            
    except Exception as e:
        print(f"加载失败: {str(e)}")
    
    return results

def load_single_model(file_path):
    """加载单个模型结果（包含等级信息）"""
    results = {}
    with open(file_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            results[data['idx']] = {
                'scores': data['score'],
                'level': data['level']
            }
    return results

def academic_visualization_v3(results, model_names=("Model A", "Model B", "Model C"), 
                             figsize=(3.5, 2.8), dpi=300):
    """优化后的等级可视化（仅柱状图）"""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 7,
        'axes.linewidth': 0.5,
        'lines.linewidth': 1,
        'lines.markersize': 2
    })
    
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # 配色方案
    # colors = ['#7FB3D5', '#F7CAC9', '#FAA460']  # 三个模型的柱颜色
    colors = ['#325ab4', '#3c8cff', '#78e6dc']
    # 数据准备（按等级排序）
    def level_sort_key(x):
        try:
            return int(x.split()[-1])  # 尝试提取数字部分
        except ValueError:
            return 5  # 非数字等级排到最后

    levels = sorted(results.keys(), key=level_sort_key)
    x = np.arange(len(levels))
    width = 0.25
    
    # 绘制柱状图
    rects1 = ax.bar(x - width, [res['m1_acc'] for res in results.values()], width, color=colors[0], label=model_names[0], edgecolor='black', linewidth=0.3)
    rects2 = ax.bar(x, [res['m2_acc'] for res in results.values()], width, color=colors[1], label=model_names[1], edgecolor='black', linewidth=0.3)
    rects3 = ax.bar(x + width, [res['m3_acc'] for res in results.values()], width, color=colors[2], label=model_names[2], edgecolor='black', linewidth=0.3)
    
    # 样式优化
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(which='both', length=0)
    
    # 坐标轴设置
    ax.set_xticks(x)
    ax.set_xticklabels(levels, fontsize=6)
    ax.set_xlabel("Difficulty Level", fontsize=7, labelpad=4)
    ax.set_ylabel("Model Accuracy", fontsize=7, labelpad=4)
    
    # 格式设置
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_ylim(0, 1)
    
    # 图例设置
    legend = ax.legend(loc='upper right', prop={'size':6})
    legend.get_frame().set_facecolor('white')
    
    return fig

# 使用示例
if __name__ == "__main__":
    # 配置模型路径
    MODEL3_PATH = "/data/qq/models/deepseek-math-7b-instruct/math_eval/original_eval/math/Recording/test_deepseek-math_seed0_t0.0_n_sample_16.jsonl"
    MODEL1_PATH = "/data/qq/models/Qwen/Qwen2.5-Math-1.5B/math_eval/original_eval/math/Recording/test_qwen_box_seed0_t0.0_n_sample_16.jsonl"
    MODEL2_PATH = "/data/qq/models/Qwen/Qwen2.5-Math-7B/math_eval/original_eval/math/Recording/test_qwen_box_seed0_t0.0_n_sample_16.jsonl"
    
    # 加载和分析数据
    analysis_results = load_level_results(MODEL1_PATH, MODEL2_PATH, MODEL3_PATH)
    
    # 生成可视化
    fig = academic_visualization_v3(
        analysis_results, 
        model_names=("Qwen2.5-Math-1.5B",   "Qwen2.5-Math-7B","Deepseek-Math-7B-Instruct"),
        figsize=(3.54, 3)
    )
    
    # 保存图表
    fig.savefig("/data/qq/data_process/plot/level_comparison_v1.png", bbox_inches='tight', pad_inches=0.1)
