import jsonlines
import random
import os
import json

# 输入输出路径
model_name = 'deepseek-math-7b-instruct'
tem = '0.7'
data_name = 'math-merge'

input_pre = f'/data/qq/models/deepseek-math-7b-instruct/math_eval/{data_name}/data/tem_{tem}'
out_pre = f'/data/qq/LLaMA-Factory/data/deepseek-math-7b-instruct/{data_name}/tem_{tem}'

input_files = [
    f'{input_pre}/level_1.jsonl',
    f'{input_pre}/level_2.jsonl',
    f'{input_pre}/level_3.jsonl'
]

output_files = [
    f'{out_pre}/stage_1.jsonl',
    f'{out_pre}/stage_2.jsonl',
    f'{out_pre}/stage_3.jsonl'
]

sample_ratios = [
    {"file_1": 0.8, "file_2": 0,   "file_3": 0},   # Stage 1 各文件比例
    {"file_1": 0.1, "file_2": 0.9, "file_3": 0},   # Stage 2 各文件比例
    {"file_1": 0.1, "file_2": 0.1, "file_3": 1}    # Stage 3 各文件比例
]

def update_data_info(new_data):
    """更新数据集信息到 dataset_info.json"""
    data_info_path = '/data/qq/LLaMA-Factory/data/dataset_info.json'
    try:
        with open(data_info_path, 'r', encoding='utf-8') as f:
            data_info = json.load(f)
    except FileNotFoundError:
        data_info = {}
    
    data_info.update(new_data)
    with open(data_info_path, 'w', encoding='utf-8') as f:
        json.dump(data_info, f, indent=2, ensure_ascii=False)
    print(f"更新 dataset_info.json: {list(new_data.keys())}")

# 创建输出目录
os.makedirs(out_pre, exist_ok=True)

# 初始化各阶段数据容器
stage_data = [[] for _ in range(3)]

# 处理每个输入文件
for file_idx, input_file in enumerate(input_files, 1):
    file_key = f"file_{file_idx}"
    print(f"处理文件: {input_file}")
    
    # 读取并打乱数据
    with jsonlines.open(input_file) as reader:
        data = list(reader)
        random.shuffle(data)
        total = len(data)
        print(f"原始数据量: {total} 条")
        
        # 获取该文件在各阶段的比例
        ratios = [stage[file_key] for stage in sample_ratios]
        assert abs(sum(ratios) - 1.0) < 1e-9, f"文件{file_key}比例总和不为1"
        
        # 计算分割点
        split_points, current = [], 0
        for ratio in ratios[:-1]:
            current += int(ratio * total)
            split_points.append(current)
        split_points.append(total)  # 确保最后阶段包含剩余数据
        
        # 分割数据到各阶段
        prev = 0
        for stage_idx, point in enumerate(split_points):
            stage_part = data[prev:point]
            stage_data[stage_idx].extend(stage_part)
            print(f"分配 {len(stage_part)} 条到 Stage {stage_idx+1}")
            prev = point

# 写入各阶段数据并更新配置
for stage_idx in range(3):
    # 写入数据
    output_path = output_files[stage_idx]
    with jsonlines.open(output_path, 'w') as writer:
        writer.write_all(stage_data[stage_idx])
    print(f"Stage {stage_idx+1} 写入完成，数据量: {len(stage_data[stage_idx])}")
    
    # 生成配置信息
    dataset_key = f"{data_name}_{model_name}_stage_{stage_idx+1}_tem_{tem}"
    new_entry = {
        dataset_key: {
            "file_name": os.path.relpath(output_path, '/data/qq/LLaMA-Factory/data'),
            "columns": {"prompt": "question", "response": "solution"}
        }
    }
    update_data_info(new_entry)

print("全部分阶段数据处理完成！")