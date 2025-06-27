import matplotlib.pyplot as plt
import numpy as np

# 百分比（总和为 1）
region_ratios = {
    "Deepseek": 0.047,
    "Qwen-7B": 0.123,
    "Qwen-1.5B": 0.058,
    "Deepseek ∩ Qwen-7B": 0.171,
    "Deepseek ∩ Qwen-1.5B": 0.031,
    "Qwen-7B ∩ Qwen-1.5B": 0.115,
    "All": 0.196,
    "None": 0.259
}

total_points = 1000

# 替换后的模型中心
centers = {
    'Deepseek': (2.05, 2.5),
    'Qwen-7B': (3.2, 2.1),
    'Qwen-1.5B': (2.95, 3.55)
}
radius = 1.5

# 颜色
colors = {
    "Deepseek": "blue",
    "Qwen-7B": "red",
    "Qwen-1.5B": "green",
    "Deepseek ∩ Qwen-7B": "purple",
    "Deepseek ∩ Qwen-1.5B": "cyan",
    "Qwen-7B ∩ Qwen-1.5B": "orange",
    "All": "black",
    "None": "gray"
}

def generate_points(condition, n_points):
    points = []
    while len(points) < n_points:
        x, y = np.random.uniform(0, 6, 2)
        in_A = (x - centers['Deepseek'][0])**2 + (y - centers['Deepseek'][1])**2 <= radius**2
        in_B = (x - centers['Qwen-7B'][0])**2 + (y - centers['Qwen-7B'][1])**2 <= radius**2
        in_C = (x - centers['Qwen-1.5B'][0])**2 + (y - centers['Qwen-1.5B'][1])**2 <= radius**2
        if condition(in_A, in_B, in_C):
            points.append([x, y])
    return np.array(points)

# 画布与样式设置
plt.figure(figsize=(14, 10))
ax = plt.gca()
ax.set_xticks([])
ax.set_yticks([])
ax.tick_params(axis='both', which='both', length=0)
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_edgecolor('black')
    spine.set_linewidth(0.8)
ax.grid(True, linestyle='--', alpha=0.3)

# 绘制点
for region, ratio in region_ratios.items():
    n = int(total_points * ratio)
    if region == "Deepseek":
        condition = lambda A, B, C: A and not B and not C
    elif region == "Qwen-7B":
        condition = lambda A, B, C: B and not A and not C
    elif region == "Qwen-1.5B":
        condition = lambda A, B, C: C and not A and not B
    elif region == "Deepseek ∩ Qwen-7B":
        condition = lambda A, B, C: A and B and not C
    elif region == "Deepseek ∩ Qwen-1.5B":
        condition = lambda A, B, C: A and C and not B
    elif region == "Qwen-7B ∩ Qwen-1.5B":
        condition = lambda A, B, C: B and C and not A
    elif region == "All":
        condition = lambda A, B, C: A and B and C
    elif region == "None":
        condition = lambda A, B, C: not A and not B and not C

    points = generate_points(condition, n)
    plt.scatter(points[:, 0], points[:, 1], color=colors[region], 
                label=f"{region} ({ratio*100:.1f}%)", alpha=0.6, s=30)

# 绘制三个模型的集合边界
theta = np.linspace(0, 2*np.pi, 100)
for center in centers.values():
    x_circle = center[0] + radius * np.cos(theta)
    y_circle = center[1] + radius * np.sin(theta)
    plt.plot(x_circle, y_circle, 'k--', linewidth=1, alpha=0.5)
# 在调用legend之前打印当前字体设置
print(plt.rcParams['legend.fontsize'])
# 图例
plt.legend(bbox_to_anchor=(0.68, 0.98), loc='upper left', 
            frameon=True, prop={'size':11,'weight': 'bold'})

# 保存与显示
plt.savefig('scatter_venn_named.pdf', bbox_inches='tight', dpi=300)
plt.show()
