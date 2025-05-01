import json
import matplotlib.pyplot as plt

# 初始化字典来存储每个类别的正确回答数和总问题数
level_correct = {"level 1": 0, "level 2": 0, "level 3": 0}
level_total = {"level 1": 0, "level 2": 0, "level 3": 0}
stage = 'base'
# 读取 JSONL 文件
with open('/data/qq/models/Qwen/Qwen2-1.5B-Instruct/math_eval/Recording/test_direct_seed0_t0.0_n_sample_1.jsonl', 'r') as f:
    for line in f:
        # 每行是一个JSON对象
        data = json.loads(line)
        
        # 获取 level 和 score
        level = data['test_level']
        score = data['score']
        
        # 计算当前问题的正确回答数
        correct_count = sum(score)
        total_count = len(score)
        
        # 根据 level 对问题进行分类
        if level == 0:
            level = "level 1"
        elif level in [1, 2, 3]:
            level = "level 2"
        else:  # level == 4
            level = "level 3"
        
        # 更新字典中的计数
        level_correct[level] += correct_count
        level_total[level] += total_count

# 提取分类、正确回答数和总问题数
categories = list(level_correct.keys())
correct_counts = list(level_correct.values())
total_counts = list(level_total.values())

# 计算每个类别的正确比例
print(correct_counts)
print(total_counts)
correct_ratios = [correct / total for correct, total in zip(correct_counts, total_counts)]

# 绘制堆叠柱状图
fig, ax = plt.subplots(figsize=(8, 5))

# 创建堆叠柱状图：先绘制总问题数的底部（蓝色），然后绘制正确答案数的顶部（红色）
ax.bar(categories, total_counts, color='blue', label='Total Questions')
ax.bar(categories, correct_counts, color='red', label='Correct Answers')

# 显示每个类别的正确比例
for i, level in enumerate(categories):
    ax.text(i, total_counts[i] + 0.1, f'{correct_ratios[i]*100:.2f}%', ha='center', va='bottom', color='black')

# 添加标题和标签
ax.set_xlabel('Level')
ax.set_ylabel('Number of Answers')
ax.set_title(f'Acc per Level Stage_{stage}')
ax.legend()

# 保存图表为文件（例如保存为PNG格式）
plt.tight_layout()
plt.savefig('/data/qq/models/Qwen/Qwen2-1.5B-Instruct/math_eval/Recording/level_correct_answers.png')  # 保存为PNG文件

# 如果你想保存为PDF文件，可以使用:
# plt.savefig('level_correct_answers.pdf')

# 显示图表
plt.show()