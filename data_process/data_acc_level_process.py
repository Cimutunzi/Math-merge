import jsonlines
import os
import json

# 设置文件路径
input_file = '/data/qq/models/Qwen/Qwen2.5-Math-7B/math_eval/math-merge/Recording/train_qwen_box_seed0_t0.7_n_sample_16.jsonl'  # 输入文件路径
out_pre = f'/data/qq/models/Qwen/Qwen2.5-Math-7B/math_eval/math-merge/data/tem_0.7'

output_file_1 = f'{out_pre}/level_1.jsonl'  # 输出文件路径
output_file_2 = f'{out_pre}/level_2.jsonl'  # 输出文件路径
output_file_3 = f'{out_pre}/level_3.jsonl'  # 输出文件路径

# 定义准确率区间
level_boundaries = {
    "level_3": (0, 20),
    "level_2": (20, 70),
    "level_1": (70, 100)
}

# 用于存储符合条件的数据
level_1_data = []
level_2_data = []
level_3_data = []

# 读取数据并处理
with jsonlines.open(input_file) as reader:
    # 遍历文件中的每一行数据
    for obj in reader:
        score_array = obj['score']
        total_elements = len(score_array)  # 数组长度
        true_count = score_array.count(True)  # True 的数量
        
        # 计算准确率（True 的比例）
        true_percentage = (true_count / total_elements) * 100 if total_elements else 0
        
        # 根据准确率将数据分配到对应的等级
        # if level_boundaries["level_1"][0] <= true_percentage < level_boundaries["level_1"][1]:
        #     level_1_data.append({ 'idx': obj['idx'], 'question': obj['question'], 'answer': obj['answer'], 'level': obj['test_level'], 'accuracy': true_percentage })
        # elif level_boundaries["level_2"][0] <= true_percentage < level_boundaries["level_2"][1]:
        #     level_2_data.append({ 'idx': obj['idx'], 'question': obj['question'], 'answer': obj['answer'], 'level': obj['test_level'], 'accuracy': true_percentage })
        # elif level_boundaries["level_3"][0] <= true_percentage < level_boundaries["level_3"][1]:
        #     level_3_data.append({ 'idx': obj['idx'], 'question': obj['question'], 'answer': obj['answer'], 'level': obj['test_level'], 'accuracy': true_percentage })
        
        if level_boundaries["level_1"][0] <= true_percentage <= level_boundaries["level_1"][1]:
            level_1_data.append({ 'idx': obj['idx'], 'question': obj['question'], 'answer': obj['solution'], 'level': obj['test_level'], 'accuracy': true_percentage })
        elif level_boundaries["level_2"][0] < true_percentage < level_boundaries["level_2"][1]:
            level_2_data.append({ 'idx': obj['idx'], 'question': obj['question'], 'answer': obj['solution'], 'level': obj['test_level'], 'accuracy': true_percentage })
        elif level_boundaries["level_3"][0] <= true_percentage < level_boundaries["level_3"][1]:
            level_3_data.append({ 'idx': obj['idx'], 'question': obj['question'], 'answer': obj['solution'], 'level': obj['test_level'], 'accuracy': true_percentage })

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