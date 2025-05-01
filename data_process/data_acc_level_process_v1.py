import jsonlines
import os
import json

# 设置文件路径
input_file = '/data/qq/models/deepseek-math-7b-instruct/math_eval/math-merge/Recording/train_deepseek-math_seed0_t0.7_n_sample_16.jsonl'  # 输入文件路径
out_pre = f'/data/qq/models/deepseek-math-7b-instruct/math_eval/math-merge/data/tem_0.7'

output_file_1 = f'{out_pre}/level_1.jsonl'  # 输出文件路径
output_file_2 = f'{out_pre}/level_2.jsonl'  # 输出文件路径
output_file_3 = f'{out_pre}/level_3.jsonl'  # 输出文件路径

# 设置数据划分比例（3:4:5），您可以在这里轻松修改
split_ratio = [1, 1, 1]

# 用于存储符合条件的数据
all_data = []

# 读取数据并处理
with jsonlines.open(input_file) as reader:
    # 遍历文件中的每一行数据
    for obj in reader:
        score_array = obj['score']
        total_elements = len(score_array)  # 数组长度
        true_count = score_array.count(True)  # True 的数量
        
        # 计算准确率（True 的比例）
        true_percentage = (true_count / total_elements) * 100 if total_elements else 0
        
        # 将数据添加到列表
        all_data.append({ 
            'idx': obj['idx'], 
            'question': obj['question'], 
            'solution': obj['solution'], 
            'level': obj['test_level'], 
            'accuracy': true_percentage 
        })

# 对数据按准确率排序
sorted_data = sorted(all_data, key=lambda x: x['accuracy'], reverse=True)

# 计算分割比例
total_count = len(sorted_data)
total_parts = sum(split_ratio)  # 总的比例（例如，3 + 4 + 5 = 12）
level_counts = [int((part / total_parts) * total_count) for part in split_ratio]

# 根据比例切分数据
level_1_data = sorted_data[:level_counts[0]]
level_2_data = sorted_data[level_counts[0]:level_counts[0] + level_counts[1]]
level_3_data = sorted_data[level_counts[0] + level_counts[1]:]

# 如果输出目录不存在，则创建
if not os.path.exists(out_pre):
    os.makedirs(out_pre)

# 将筛选后的数据写入新文件
with jsonlines.open(output_file_1, mode='w') as writer:
    writer.write_all(level_1_data)

with jsonlines.open(output_file_2, mode='w') as writer:
    writer.write_all(level_2_data)

with jsonlines.open(output_file_3, mode='w') as writer:
    writer.write_all(level_3_data)

print(f"数据处理完成，{len(level_1_data)}条level_1数据保存至 {output_file_1}")
print(f"数据处理完成，{len(level_2_data)}条level_2数据保存至 {output_file_2}")
print(f"数据处理完成，{len(level_3_data)}条level_3数据保存至 {output_file_3}")
