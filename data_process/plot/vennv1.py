import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge

# 加载模型结果
def load_model_results(file_path):
    results = {}
    with open(file_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            idx = data['idx']
            score = data['score'][0] if isinstance(data['score'], list) else data['score']
            results[idx] = score
    return results

# 模型文件路径
model_a = load_model_results('/data/qq/models/deepseek-math-7b-instruct/math_eval/original_eval/math-merge/Recording/train_deepseek-math_seed0_t0.0_n_sample_1.jsonl')
model_b = load_model_results('/data/qq/models/Qwen/Qwen2.5-Math-7B/math_eval/original_eval/math-merge/Recording/train_qwen_box_seed0_t0.0_n_sample_1.jsonl')
model_c = load_model_results('/data/qq/models/Qwen/Qwen2.5-Math-1.5B/math_eval/original_eval/math-merge/Recording/train_qwen_box_seed0_t0.0_n_sample_1.jsonl')

# 分类标签名称替换
category_counts = {
    'Deepseek_only': 0, 'Qwen7B_only': 0, 'Qwen1.5B_only': 0,
    'Deepseek_Qwen7B': 0, 'Deepseek_Qwen1.5B': 0, 'Qwen7B_Qwen1.5B': 0,
    'All_correct': 0, 'All_wrong': 0
}

# 分类处理
data = []
for idx in model_a:
    a = model_a.get(idx, False)
    b = model_b.get(idx, False)
    c = model_c.get(idx, False)

    if a and b and c:
        category = 'All_correct'
    elif not a and not b and not c:
        category = 'All_wrong'
    elif a and not b and not c:
        category = 'Deepseek_only'
    elif not a and b and not c:
        category = 'Qwen7B_only'
    elif not a and not b and c:
        category = 'Qwen1.5B_only'
    elif a and b and not c:
        category = 'Deepseek_Qwen7B'
    elif a and not b and c:
        category = 'Deepseek_Qwen1.5B'
    elif not a and b and c:
        category = 'Qwen7B_Qwen1.5B'
    else:
        category = 'All_wrong'

    category_counts[category] += 1
    data.append(category)

total = len(data)

# 颜色配置
color_palette = {
    'Deepseek_only': (1.0, 0, 0),             # 红
    'Qwen7B_only': (1.0, 0.9, 0),             # 黄
    'Qwen1.5B_only': (0, 0.6, 1.0),           # 蓝
    'Deepseek_Qwen7B': (1.0, 0.5, 0),         # 橙
    'Deepseek_Qwen1.5B': (0.8, 0, 0.8),       # 紫
    'Qwen7B_Qwen1.5B': (0, 0.8, 0),           # 绿
    'All_correct': (0.4, 0.4, 0.4),           # 中灰
    'All_wrong': (0.7, 0.7, 0.7)              # 深灰
}

# 扇区设置
sector_config = {
    'Deepseek_only': {'angle': (-30, 30), 'radius': 1.4},
    'Qwen7B_only': {'angle': (90, 150), 'radius': 1.4},
    'Qwen1.5B_only': {'angle': (210, 270), 'radius': 1.4},
    'Deepseek_Qwen7B': {'angle': (30, 90), 'radius': 1.4},
    'Deepseek_Qwen1.5B': {'angle': (150, 210), 'radius': 1.4},
    'Qwen7B_Qwen1.5B': {'angle': (270, 330), 'radius': 1.4},
    'All_correct': {'radius': 0.7},
    'All_wrong': {'radius': 1.65}
}

# 坐标生成
def generate_coordinates(category):
    config = sector_config[category]

    if category == 'All_correct':
        r = np.random.uniform(0, config['radius'])
        angle = np.random.uniform(0, 360)
    elif category == 'All_wrong':
        r = np.random.uniform(config['radius']-0.2, config['radius'])
        angle = np.random.uniform(0, 360)
    else:
        angle_range = config['angle']
        angle = np.random.uniform(angle_range[0], angle_range[1])
        r = np.random.uniform(config['radius']-0.6, config['radius'])

    theta = np.radians(angle)
    return (r * np.cos(theta), r * np.sin(theta))

# 生成点
coordinates, colors = [], []
for category in data:
    coordinates.append(generate_coordinates(category))
    colors.append(color_palette[category])

x = [c[0] for c in coordinates]
y = [c[1] for c in coordinates]

# 绘图
fig, ax = plt.subplots(figsize=(12, 12), facecolor='white')
ax.scatter(x, y, c=colors, s=10, alpha=0.75, edgecolors='none')

# 标签配置
label_config = {
    'Deepseek_only': {'fontsize': 12, 'color': 'darkred', 'weight': 'bold'},
    'Qwen7B_only': {'fontsize': 12, 'color': 'darkgoldenrod', 'weight': 'bold'},
    'Qwen1.5B_only': {'fontsize': 12, 'color': 'darkblue', 'weight': 'bold'},
    'Deepseek_Qwen7B': {'fontsize': 11, 'color': 'black'},
    'Deepseek_Qwen1.5B': {'fontsize': 11, 'color': 'black'},
    'Qwen7B_Qwen1.5B': {'fontsize': 11, 'color': 'black'},
    'All_correct': {'fontsize': 12, 'color': 'dimgray'},
    'All_wrong': {'fontsize': 11, 'color': 'black', 'offset': (0, 0.5)}
}

# 添加标签
for category in sector_config:
    count = category_counts[category]
    if count == 0:
        continue
    percent = count / total * 100
    text = f"{count}\n({percent:.1f}%)"
    if category in ['All_correct', 'All_wrong']:
        x_pos, y_pos = 0, sector_config[category]['radius'] * 0.6
        if category == 'All_wrong':
            y_pos += label_config[category].get('offset', (0, 0))[1]
    else:
        angle_center = np.mean(sector_config[category]['angle'])
        r = sector_config[category]['radius'] * 0.75
        theta = np.radians(angle_center)
        x_pos = r * np.cos(theta)
        y_pos = r * np.sin(theta)

    ax.annotate(text,
                xy=(x_pos, y_pos),
                xytext=(x_pos * 1.05, y_pos * 1.05),
                ha='center',
                va='center',
                fontsize=label_config[category]['fontsize'],
                color=label_config[category]['color'],
                weight=label_config[category].get('weight', 'normal'),
                arrowprops=dict(arrowstyle="->", color='gray', alpha=0.7, linewidth=1),
                bbox=dict(facecolor='white', alpha=0.97,
                          edgecolor=label_config[category].get('color', 'gray'),
                          boxstyle='round,pad=0.3'))

# 图例
legend_elements = [
    plt.Line2D([0], [0], marker='o', color='w', label=label, markerfacecolor=color, markersize=12)
    for label, color in [
        ('Deepseek-Math-7B only', color_palette['Deepseek_only']),
        ('Qwen2.5-Math-7B only', color_palette['Qwen7B_only']),
        ('Qwen2.5-Math-1.5B only', color_palette['Qwen1.5B_only']),
        ('Deepseek & Qwen7B', color_palette['Deepseek_Qwen7B']),
        ('Deepseek & Qwen1.5B', color_palette['Deepseek_Qwen1.5B']),
        ('Qwen7B & Qwen1.5B', color_palette['Qwen7B_Qwen1.5B']),
        ('All correct', color_palette['All_correct']),
        ('All wrong', color_palette['All_wrong']),
    ]
]

ax.legend(handles=legend_elements,
          loc='upper left',
          bbox_to_anchor=(0.9, 0.95),
          title="Categories",
          frameon=False,
          fontsize=12)

ax.axis('equal')
ax.axis('off')
plt.tight_layout()
plt.savefig("final_optimized_venn_named.png", dpi=300, bbox_inches='tight', facecolor='white')
plt.show()
