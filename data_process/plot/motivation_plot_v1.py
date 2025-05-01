import json
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter

def load_level_results(m1_path, m2_path):
    """加载单个数据集的等级结果
    Args:
        m1_path: 模型1结果路径
        m2_path: 模型2结果路径
    Returns:
        dict: 按等级分组的统计结果
    """
    results = {}
    
    try:
        # 加载模型结果
        m1_results = load_single_model(m1_path)
        m2_results = load_single_model(m2_path)
        
        # 验证一致性并获取共同问题ID
        common_ids = set(m1_results.keys()) & set(m2_results.keys())
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
            both_correct = sum(1 for idx in ids if m1_results[idx]['score'] and m2_results[idx]['score'])
            only_one_correct = sum(1 for idx in ids if m1_results[idx]['score'] ^ m2_results[idx]['score'])
            
            results[level] = {
                'total': total,
                'both_correct': both_correct,
                'only_one_correct': only_one_correct,
                'm1_acc': sum(m1_results[idx]['score'] for idx in ids)/total,
                'm2_acc': sum(m2_results[idx]['score'] for idx in ids)/total
            }
            
    except Exception as e:
        print(f"加载失败: {str(e)}")
    
    return results

def academic_visualization_v3(results, model_names=("Model A", "Model B"), 
                             figsize=(3.5, 2.8), dpi=300):
    """优化后的等级可视化"""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 7,
        'axes.linewidth': 0.5,
        'lines.linewidth': 1,
        'lines.markersize': 2
    })
    
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # 配色方案
    colors = {
        'both': '#7FB3D5',    # 共同正确
        'single': '#F7CAC9',  # 仅一个正确
        'line1': '#FAA460',   # 模型1折线
        'line2': '#87CEFA'    # 模型2折线
    }
    
    # 数据准备（按等级排序）
    def level_sort_key(x):
        try:
            return int(x.split()[-1])  # 尝试提取数字部分
        except ValueError:
            print(x)
            return 5  # 非数字等级排到最后

    levels = sorted(results.keys(), key=level_sort_key)
    # levels = sorted(results.keys(), key=lambda x: int(x.split()[-1]))
    x = np.arange(len(levels))
    width = 0.28
    
    # 转换百分比
    both_pct = [res['both_correct']/res['total'] for res in results.values()]
    single_pct = [res['only_one_correct']/res['total'] for res in results.values()]
    
    # 绘制柱状图
    rects1 = ax.bar(x - width/2, both_pct, width, 
                   color=colors['both'], label='Both Correct',
                   edgecolor='black', linewidth=0.3, align='center')
    rects2 = ax.bar(x + width/2, single_pct, width, 
                   color=colors['single'], label='Only One Correct',
                   edgecolor='black', linewidth=0.3, align='center')
    
    # 绘制折线图（次要坐标轴）
    ax2 = ax.twinx()
    line1, = ax2.plot(x, [res['m1_acc'] for res in results.values()], 
                     color=colors['line1'], marker='o', 
                     markersize=2.5,
                     markeredgewidth=0.5,
                     linestyle='--',
                     label=model_names[0])
    line2, = ax2.plot(x, [res['m2_acc'] for res in results.values()], 
                     color=colors['line2'], marker='s',
                     markersize=2.3,
                     markeredgewidth=0.5,           
                     linestyle='--',
                     label=model_names[1])
    
    # 样式优化
    ax.spines['top'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    for axis in [ax, ax2]:
        axis.tick_params(which='both', length=0)
    
    # 坐标轴设置
    ax.set_xticks(x)
    ax.set_xticklabels(levels, fontsize=6)
    ax.set_xlabel("Original Difficulty Level", fontsize=7, labelpad=4)
    ax.set_ylabel("Accuracy Proportion", fontsize=7, labelpad=4)
    ax2.set_ylabel("Model Accuracy", fontsize=7, labelpad=4)
    
    # 格式设置
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax2.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_ylim(0, 1)
    ax2.set_ylim(0, 1)
    
    # 图例设置
    lines = [rects1, rects2, line1, line2]
    labels = [l.get_label() for l in lines]
    legend = ax.legend(lines, labels,
                      loc='upper left',
                      bbox_to_anchor=(0.45, 1.05),
                      frameon=True,
                      fontsize=6)
    legend.get_frame().set_facecolor('white')
    
    return fig

def load_single_model(file_path):
    """加载单个模型结果（包含等级信息）"""
    results = {}
    with open(file_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            results[data['idx']] = {
                'score': data['score'][0],
                'level': data['level']
            }
    return results

# 使用示例
if __name__ == "__main__":
    # 配置模型路径
    MODEL1_PATH = "/data/qq/models/deepseek-math-7b-instruct/math_eval/original_eval/math/Recording/test_deepseek-math_seed0_t0.0_n_sample_1.jsonl"
    MODEL2_PATH = "/data/qq/models/Qwen/Qwen2.5-Math-1.5B/math_eval/original_eval/math/Recording/test_qwen_box_seed0_t0.0_n_sample_1.jsonl"
    
    # 加载和分析数据
    analysis_results = load_level_results(MODEL1_PATH, MODEL2_PATH)
    
    # 生成可视化
    fig = academic_visualization_v3(
        analysis_results, 
        model_names=("deepseek-math-7b-instruct", "Qwen2.5-Math-1.5B"),
        figsize=(3.54, 3)
    )
    
    # 保存图表
    fig.savefig("/data/qq/data_process/plot/level_comparison.pdf", bbox_inches='tight', pad_inches=0.1)