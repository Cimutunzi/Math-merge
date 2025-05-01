import json
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter

def load_dataset_results(dataset_info):
    """加载多个数据集的评测结果
    Args:
        dataset_info: 字典格式 {数据集名称: (model1_path, model2_path)}
    Returns:
        dict: 结构化的统计结果
    """
    results = {}
    
    for ds_name, (m1_path, m2_path) in dataset_info.items():
        try:
            # 加载模型结果
            m1_results = load_single_model(m1_path)
            m2_results = load_single_model(m2_path)
            
            # 验证一致性
            common_ids = set(m1_results.keys()) & set(m2_results.keys())
            if not common_ids:
                print(f"数据集 {ds_name} 无共同问题ID，跳过")
                continue
                
            # 统计指标
            total = len(common_ids)
            both_correct = sum(1 for idx in common_ids if m1_results[idx] and m2_results[idx])
            only_one_correct = sum(1 for idx in common_ids if m1_results[idx] ^ m2_results[idx])
            m1_acc = sum(m1_results[idx] for idx in common_ids) / total
            m2_acc = sum(m2_results[idx] for idx in common_ids) / total
            
            results[ds_name] = {
                'total': total,
                'both_correct': both_correct,
                'only_one_correct': only_one_correct,
                'm1_acc': m1_acc,
                'm2_acc': m2_acc
            }
            
        except Exception as e:
            print(f"加载数据集 {ds_name} 失败: {str(e)}")
    
    return results

def academic_visualization_v3(results, model_names=("Model A", "Model B"), 
                             figsize=(3.5, 2.8), dpi=300):
    """最终优化版本：解决标签重叠问题"""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 7,
        'axes.linewidth': 0.5,
        'lines.linewidth': 1,
        'lines.markersize': 2
    })
    
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # 配色方案保持不变
    colors = {
        'both': '#7FB3D5',    # 保持原浅蓝柱状
        'single': '#F7CAC9',  # 保持原浅粉柱状
        'line1': '#FAA460',   # 深蓝灰（原色替代方案）
        'line2': '#87CEFA'    # 蓝紫色
    }
    
    
    # 数据准备
    datasets = [d.replace("QA", "\nQA") for d in results.keys()]  # 添加换行符
    x = np.arange(len(datasets))
    width = 0.28
    
    # 转换百分比
    both_pct = [res['both_correct']/res['total'] for res in results.values()]
    single_pct = [res['only_one_correct']/res['total'] for res in results.values()]
    
    # 绘制柱状图
    rects1 = ax.bar(x - width/2, both_pct, width, 
                   color=colors['both'], label='Both Correct',
                   edgecolor='black', linewidth=0.3,align='center')
    rects2 = ax.bar(x + width/2, single_pct, width, 
                   color=colors['single'], label='Only One Correct',
                   edgecolor='black', linewidth=0.3,align='center')
    
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
    
    # ====== 关键修改部分 ======    
    # 1. 移除顶部边框线
    ax.spines['top'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    
    # 2. 隐藏所有刻度线
    for axis in [ax, ax2]:
        axis.tick_params(
            which='both',
            length=0,          # 刻度线长度设为0
            top=False,         # 顶部刻度不显示
            bottom=False,      # 底部刻度线隐藏
            left=False,
            right=False
        )
    
    # 3. 保留底部轴线（无刻度）
    ax.spines['bottom'].set_visible(True)
    
    # ====== 坐标轴优化 ======    
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, 
                    rotation=0,          # 减小旋转角度
                    ha='center',
                    rotation_mode='anchor',          # 水平居中
                    fontsize=6)           # 加粗横坐标标签
    
    # Y轴设置
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_ylim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.yaxis.set_major_formatter(PercentFormatter(1.0))
    
    ax.set_xlabel("Dataset", fontsize=7, labelpad=4)  # X轴说明
    ax.set_ylabel("Accuracy Proportion", fontsize=7, labelpad=4)  # 左Y轴
    ax2.set_ylabel("Model Accuracy", fontsize=7, labelpad=4)  # 右Y轴

    # 网格线优化
    ax.yaxis.grid(True, linestyle=':', linewidth=0.3)
    
    # 图例布局调整
    lines = [rects1, rects2, line1, line2]
    labels = [l.get_label() for l in lines]
    
    legend = ax.legend(lines, labels,
                      loc='upper left',
                      bbox_to_anchor=(0.45, 1),
                      frameon=True,
                      framealpha=0.9,
                      edgecolor='#FFFFFF',
                      ncol=1,
                      handlelength=1.0,
                      fontsize=6,
                      borderaxespad=0.5)
    
    # 设置图例背景
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_linewidth(0.3)
    
    # 调整图表布局，增加顶部和底部的间距
    # plt.subplots_adjust(top=0.8, bottom=0.2)  # 调整上下边距，确保不挤压
    # plt.tight_layout(pad=0.8)
    return fig


# 辅助函数
def load_single_model(file_path):
    """加载单个模型结果"""
    results = {}
    with open(file_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            results[data['idx']] = data['score'][0]
    return results

# 使用示例
if __name__ == "__main__":
    # 配置数据集信息
    DATASETS = {
        "Gsm8k": ("/data/qq/models/deepseek-math-7b-instruct/math_eval/original_eval/gsm8k/Recording/test_deepseek-math_seed0_t0.0_n_sample_1.jsonl", "/data/qq/models/Qwen/Qwen2.5-Math-1.5B/math_eval/original_eval/gsm8k/Recording/test_qwen_box_seed0_t0.0_n_sample_1.jsonl"),
        "Minerva-math": ("/data/qq/models/deepseek-math-7b-instruct/math_eval/original_eval/minerva-math/Recording/test_deepseek-math_seed0_t0.0_n_sample_1.jsonl", "/data/qq/models/Qwen/Qwen2.5-Math-1.5B/math_eval/original_eval/minerva-math/Recording/test_qwen_box_seed0_t0.0_n_sample_1.jsonl"),
        "Math-500": ("/data/qq/models/deepseek-math-7b-instruct/math_eval/original_eval/MATH-500/Recording/test_deepseek-math_seed0_t0.0_n_sample_1.jsonl", "/data/qq/models/Qwen/Qwen2.5-Math-1.5B/math_eval/original_eval/MATH-500/Recording/test_qwen_box_seed0_t0.0_n_sample_1.jsonl"),
        "Olympiadbench": ("/data/qq/models/deepseek-math-7b-instruct/math_eval/original_eval/olympiadbench/Recording/test_deepseek-math_seed0_t0.0_n_sample_1.jsonl", "/data/qq/models/Qwen/Qwen2.5-Math-1.5B/math_eval/original_eval/olympiadbench/Recording/test_qwen_box_seed0_t0.0_n_sample_1.jsonl"),
    }
    
    # 加载和分析数据
    analysis_results = load_dataset_results(DATASETS)
    
    # 生成可视化
    fig = academic_visualization_v3(
        analysis_results, 
        model_names=("deepseek-math-7b-instruct", "Qwen2.5-Math-1.5B"),
        figsize=(3.54, 3)  # 对应期刊单栏宽度 (8.5cm)
    )
    
    # 保存为矢量图
    fig.savefig("/data/qq/data_process/plot/academic_plot.png", bbox_inches='tight', pad_inches=0.05)
    # fig.savefig("academic_plot.eps", format='eps', bbox_inches='tight')