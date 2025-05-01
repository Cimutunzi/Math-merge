import jsonlines
import random
import os
# 设置文件路径
input_file = '/data/qq/models/Qwen/Qwen2-1.5B-Instruct/math_eval/Recording/train_direct_seed0_t0.3_n_sample_64.jsonl'  # 输入文件路径
out_pre = f'/data/qq/models/Qwen/Qwen2-1.5B-Instruct/math_eval/gsm8k/data/tem_0.3'
output_file_1 = f'{out_pre}/level_1.jsonl'  # 输出文件路径
output_file_2 = f'{out_pre}/level_2.jsonl'  # 输出文件路径
output_file_3 = f'{out_pre}/level_3.jsonl'  # 输出文件路径

# 用于存储符合条件的数据
filtered_data = []

# 读取数据并处理
with jsonlines.open(input_file) as reader:
    level_1_data = []
    level_2_data = []
    level_3_data = []
    num = 0
    # 遍历文件中的每一行数据
    for obj in reader:
        # 提取所需的字段
        if obj['test_level'] in [0]:
            level_1_data.append({ 'idx': obj['idx'], 'question': obj['question'], 'answer': obj['answer'], 'level': obj['test_level'] })
        if obj['test_level'] in [1,2,3]:
            level_2_data.append({ 'idx': obj['idx'], 'question': obj['question'], 'answer': obj['answer'], 'level': obj['test_level'] })
        if obj['test_level'] in [4]:
            level_3_data.append({ 'idx': obj['idx'], 'question': obj['question'], 'answer': obj['answer'], 'level': obj['test_level'] })
        # if obj['test_level'] in [0]:
            # level_1_data.append({ 'idx': obj['idx'], 'problem': obj['question'], 'solution': obj['solution'], 'level': obj['test_level'] })
        #     num += 1
        # if num >= 1450:
        #     break

# 取level为1的50%数据
# sampled_level_1_data = random.sample(level_1_data, k=int(len(level_1_data) * 0.5))

# 合并level为2的所有数据与level为1的50%数据
# filtered_data.extend(sampled_level_1_data)
# filtered_data.extend(level_1_data)
# filtered_data.extend(level_2_data)

# 将筛选后的数据写入新文件
if not os.path.exists(out_pre):
    os.makedirs(out_pre)
    
with jsonlines.open(output_file_1, mode='w') as writer:
    writer.write_all(level_1_data)

with jsonlines.open(output_file_2, mode='w') as writer:
    writer.write_all(level_2_data)

with jsonlines.open(output_file_3, mode='w') as writer:
    writer.write_all(level_3_data)

print(f"数据处理完成，保存至 {output_file_1}")