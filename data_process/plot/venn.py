

import json
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# === Step 1: 加载模型分数 ===
def load_scores(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return {json.loads(line)['idx']: json.loads(line)['score'][0] for line in f}

a_scores = load_scores('/data/qq/models/deepseek-math-7b-instruct/math_eval/original_eval/math-merge/Recording/train_deepseek-math_seed0_t0.0_n_sample_1.jsonl')
b_scores = load_scores('/data/qq/models/Qwen/Qwen2.5-Math-7B/math_eval/original_eval/math-merge/Recording/train_qwen_box_seed0_t0.0_n_sample_1.jsonl')
c_scores = load_scores('/data/qq/models/Qwen/Qwen2.5-Math-1.5B/math_eval/original_eval/math-merge/Recording/train_qwen_box_seed0_t0.0_n_sample_1.jsonl')

# === Step 2: 构造答题组合类别（000 ~ 111） ===
classified_points = defaultdict(list)
for idx in a_scores.keys():
    a = int(a_scores[idx])
    b = int(b_scores[idx])
    c = int(c_scores[idx])
    code = (a << 2) | (b << 1) | c
    classified_points[code].append(idx)

# === Step 3: 配色（RGB）===
label_to_rgb = {
    0b000: (0, 0, 0),       # 黑
    0b001: (0, 0, 1),       # 蓝
    0b010: (1, 1, 0),       # 黄
    0b011: (0, 1, 1),       # 青
    0b100: (1, 0, 0),       # 红
    0b101: (1, 0, 1),       # 品红
    0b110: (1, 0.5, 0),     # 橙
    0b111: (1, 1, 1),       # 白
}

# === Step 4: 布局位置（圆环） ===
# 中心: 111
# 内环顺序：100, 110, 010, 011, 001, 101
# 外层（最远）：000

layout_order = [
    0b111,
    0b100, 0b110, 0b010, 0b011, 0b001, 0b101,
    0b000
]

layout_pos = {}
radius_level = {
    0b111: 0,
    0b100: 1, 0b110: 1, 0b010: 1, 0b011: 1, 0b001: 1, 0b101: 1,
    0b000: 2
}

# 设置位置：以圆环方式放置
angle_map = {
    0b111: 0,
    0b100: 0,
    0b110: np.pi / 3,
    0b010: 2 * np.pi / 3,
    0b011: np.pi,
    0b001: 4 * np.pi / 3,
    0b101: 5 * np.pi / 3,
    0b000: 0
}

x_all, y_all, color_all, label_xs, label_ys, label_texts = [], [], [], [], [], []

# 圆心间距
base_radius = 10

for label in layout_order:
    points = classified_points.get(label, [])
    count = len(points)
    r_layout = base_radius * radius_level[label]
    theta = angle_map[label]
    cx = r_layout * np.cos(theta)
    cy = r_layout * np.sin(theta)
    
    # 点在本类圆内均匀分布
    local_r = np.sqrt(count) / 1.8  # 控制区域大小
    for _ in range(count):
        angle = np.random.uniform(0, 2*np.pi)
        rr = local_r * np.sqrt(np.random.uniform(0, 1))
        x = cx + rr * np.cos(angle)
        y = cy + rr * np.sin(angle)
        x_all.append(x)
        y_all.append(y)
        color_all.append(label_to_rgb[label])

    # 标签
    label_xs.append(cx)
    label_ys.append(cy)
    label_texts.append(f"{label:03b} ({count})")

# === Step 5: 绘图 ===
plt.figure(figsize=(10, 10))
plt.scatter(x_all, y_all, c=color_all, s=10, alpha=0.8)

# 添加标签
for x, y, text in zip(label_xs, label_ys, label_texts):
    plt.text(x, y, text, ha='center', va='center',
             fontsize=10, bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.3'))

plt.title("Venn-Style Circular Scatter Plot (Grouped by Model Agreement)", fontsize=14)
plt.axis('off')
plt.tight_layout()
plt.show()

plt.savefig("improved_proportional_venn.png", dpi=300, bbox_inches='tight')
plt.show()
