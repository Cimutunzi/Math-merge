# import json
# import matplotlib.pyplot as plt
# import seaborn as sns
# import os

# # 用于存储每行的准确率
# accuracies = []

# # 逐行读取 JSONL 文件
# with open('/data/qq/models/deepseek-math-7b-base/math_eval/math-merge/Recording/cot_3-4_deepseek-math_seed0_t0.7_n_sample_16.jsonl', 'r') as f:
#     for line in f:
#         entry = json.loads(line.strip())  # 解析每一行 JSON 数据
        
#         score_array = entry['score']
#         total_elements = len(score_array)  # 数组长度
#         true_count = score_array.count(True)  # True 的数量
        
#         # 计算 true 占比
#         true_percentage = (true_count / total_elements) * 100 if total_elements else 0
#         accuracies.append(true_percentage)


# plt.figure(figsize=(10, 6))
# ax = sns.histplot(accuracies, kde=False, color='b', bins=20)
# # 在每个柱子上方显示频数
# for p in ax.patches:
#     height = p.get_height()  # 获取柱子的高度
#     width = p.get_width()  # 获取柱子的宽度
#     x = p.get_x()  # 获取柱子的 x 坐标
#     y = p.get_y()  # 获取柱子的 y 坐标   
#     # 在柱子上方显示频数
#     ax.text(x + width / 2, height, f'{int(height)}', ha='center', va='bottom', fontsize=10)
# # 设置标题和标签
# plt.title('Accuracy Distribution (Histogram)', fontsize=16)
# plt.xlabel('Accuracy (%)', fontsize=12)
# plt.ylabel('Frequency', fontsize=12)



# save_path = '/data/qq/models/deepseek-math-7b-base/math_eval/math-merge/data/tem_0.7/cot_3-4_accuracy_distribution_0.7_1.png'
# if not os.path.exists(save_path):
#     os.makedirs(os.path.dirname(save_path), exist_ok=True)
#     # 保存图像到文件
# plt.tight_layout()  # 优化布局
# plt.savefig(save_path, dpi=300)  # 保存为 PNG 文件，分辨率为 300dpi
#     # 显示图形（如果需要）
# plt.show()


import json
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

accuracies = []

with open('/data/qq/models/Qwen/Qwen2.5-Math-1.5B/math_eval/math-merge/Recording/train_qwen_box_seed0_t0.7_n_sample_16.jsonl', 'r') as f:
    for line in f:
        entry = json.loads(line.strip())
        score_array = entry['score']
        total_elements = len(score_array)
        true_count = score_array.count(True)
        true_percentage = (true_count / total_elements) * 100 if total_elements else 0
        accuracies.append(true_percentage)

plt.figure(figsize=(10, 6))
# 明确指定 bin 边界：从 0 到 100，每 5 一段
ax = sns.histplot(accuracies, bins=np.arange(0, 105, 10), kde=False, color='b', edgecolor='black')

plt.xlim(0, 100)
plt.xticks(np.arange(0, 105, 10))

for p in ax.patches:
    height = p.get_height()
    width = p.get_width()
    x = p.get_x()
    ax.text(x + width / 2, height, f'{int(height)}', ha='center', va='bottom', fontsize=10)

plt.title('Accuracy Distribution (Histogram)', fontsize=16)
plt.xlabel('Accuracy (%)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)

save_path = '/data/qq/models/Qwen/Qwen2.5-Math-1.5B/math_eval/math-merge/data/tem_0.7/accuracy_distribution_0.7.png'
os.makedirs(os.path.dirname(save_path), exist_ok=True)
plt.tight_layout()
plt.savefig(save_path, dpi=300)
plt.show()

