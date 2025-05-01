import jsonlines
import random
import os
import json

# 输入输出路径
# model_name = 'qwen2_1.5b_ins'
# model_name = 'qwen2_7b_ins'
# model_name = 'llama3_8b_ins'
model_name = 'deepseek-math-7b-instruct'
# model_name = 'Qwen2.5-Math-7B'
# model_name = 'Qwen2.5-Math-1.5B'


tem = '0.7'
# tem = '0.5'


data_name = 'math-merge'



# input_pre = f'/data/qq/models/Qwen/Qwen2.5-Math-1.5B/math_eval/{data_name}/data/tem_{tem}'
# out_pre = f'/data/qq/LLaMA-Factory/data/Qwen2.5-Math-1.5B/{data_name}/tem_{tem}'
input_pre = f'/data/qq/models/deepseek-math-7b-instruct/math_eval/{data_name}/data/tem_{tem}'
out_pre = f'/data/qq/LLaMA-Factory/data/deepseek-math-7b-instruct/{data_name}/tem_{tem}'

# 输入文件路径
input_file_1 = f'{input_pre}/level_1.jsonl'
input_file_2 = f'{input_pre}/level_2.jsonl'
input_file_3 = f'{input_pre}/level_3.jsonl'

# 输出文件路径
output_file  = [
    f'{out_pre}/stage_1.jsonl',
    f'{out_pre}/stage_2.jsonl',
    f'{out_pre}/stage_3.jsonl'
]

# 设定每个文件的采样比例（0~1之间）

sample_ratios = [
    {
        "file_1": 0.8,  
        "file_2": 0,  
        "file_3": 0  
    },
    {
        "file_1": 0.1,  
        "file_2": 0.9,  
        "file_3": 0  
    },
    {
        "file_1": 0.1,  
        "file_2": 0.1,  
        "file_3": 1  
    }
]

# sample_ratios = [
#     {
#         "file_1": 1,  
#         "file_2": 0,  
#         "file_3": 0  
#     },
#     {
#         "file_1": 0,  
#         "file_2": 1,  
#         "file_3": 0  
#     },
#     {
#         "file_1": 0,  
#         "file_2": 0,  
#         "file_3": 1  
#     }
# ]

# 存储采样后的数据
def sample_data(file_path, ratio):
    sampled = []
    with jsonlines.open(file_path) as reader:
        data = list(reader)  # 先将文件内容全部读取到内存中
        sample_size = int(len(data) * ratio)  # 计算采样数量
        sampled = random.sample(data, sample_size)  # 随机采样
    print(f"从 {file_path} 采样 {sample_size} 条数据")
    return sampled

# 更新 data_info.json 文件的功能
def update_data_info(new_data):
    data_info_path = '/data/qq/LLaMA-Factory/data/dataset_info.json'
    
    # 读取现有的 data_info.json 文件
    try:
        with open(data_info_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
    except FileNotFoundError:
        data = {}

    # 更新或插入新的数据集信息
    for key, value in new_data.items():
        # 如果该条数据已存在，直接覆盖
        data[key] = value

    # 将更新后的数据写回到 data_info.json 文件
    with open(data_info_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=2)

    print("数据集信息已成功更新到 data_info.json 文件")

# 创建输出目录
if not os.path.exists(out_pre):
    os.makedirs(out_pre)

# 遍历每个采样比例并进行数据处理
for i in range(len(sample_ratios)):
    filtered_data = []
    # 读取和采样每个文件
    filtered_data.extend(sample_data(input_file_1, sample_ratios[i]["file_1"]))
    filtered_data.extend(sample_data(input_file_2, sample_ratios[i]["file_2"]))
    filtered_data.extend(sample_data(input_file_3, sample_ratios[i]["file_3"]))

    # 将采样后的数据写入输出文件
    with jsonlines.open(output_file[i], mode='w') as writer:
        for data in filtered_data:
            writer.write(data)

    print(f"数据采样完成，共写入 {len(filtered_data)} 条数据到 {output_file[i]}")

    # 生成新的数据集条目
    new_data = {
        f"{data_name}_{model_name}_stage_{i+1}_tem_{tem}": {
            "file_name": os.path.relpath(output_file[i], '/data/qq/LLaMA-Factory/data'),
            "columns": {
                "prompt": "question",
                "response": "solution"
            }
        }
    }

    # 更新 data_info.json
    update_data_info(new_data)