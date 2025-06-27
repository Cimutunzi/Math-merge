import jsonlines
import random
import os
import json

# 配置参数
model_name = 'deepseek-math-7b-instruct'
tem = '0.7'
data_name = 'math-merge'

# 路径配置
input_pre = '/data/qq/data/math-merge/data/deepseek-math-7b-instruct'
out_pre = f'/data/qq/data/math-merge/data/{model_name}/'

# 文件配置
input_files = [
    f'{input_pre}/level_1.jsonl',
    f'{input_pre}/level_2.jsonl',
    f'{input_pre}/level_3-hint.jsonl'
]

output_files = [
    f'{out_pre}/stage_1.jsonl',
    f'{out_pre}/stage_2.jsonl',
    f'{out_pre}/stage_3.jsonl'
]

# 比例配置（总和可以≤1）
sample_ratios = [
    {  # Stage 1
        "file_1": 0.8,  # level_1的80%
        "file_2": 0.0,  # level_2不参与
        "file_3": 0.0   # level_3不参与
    },
    {  # Stage 2
        "file_1": 0.1,  # level_1的20%
        "file_2": 0.9,  # level_2的90%
        "file_3": 0.0   # level_3不参与
    },
    {  # Stage 3
        "file_1": 0.0,  # level_1不参与
        "file_2": 0.1,  # level_2的10%
        "file_3": 1   # level_3的90%（舍弃10%）
    }
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

def process_file(input_file, file_idx):
    """处理单个文件"""
    file_key = f"file_{file_idx}"
    print(f"\n处理文件: {input_file}")
    
    with jsonlines.open(input_file) as reader:
        data = list(reader)
        random.shuffle(data)
        total = len(data)
        print(f"原始数据量: {total} 条 | 可舍弃部分数据")
        
        # 获取各阶段分配比例
        ratios = [stage[file_key] for stage in sample_ratios]
        
        # 计算实际分配量
        allocated = 0
        split_points = []
        for ratio in ratios:
            alloc_num = int(ratio * total)  # 直接取整截断
            allocated += alloc_num
            split_points.append(allocated)
        
        # 显示舍弃数据
        if allocated < total:
            discard = total - allocated
            print(f"将舍弃 {discard} 条数据（{discard/total:.1%}）")
        
        # 分割数据到各阶段
        prev = 0
        for stage_idx, point in enumerate(split_points):
            if point <= prev:  # 无数据分配时跳过
                continue
            stage_part = data[prev:point]
            stage_data[stage_idx].extend(stage_part)
            print(f"Stage {stage_idx+1} 获得 {len(stage_part)} 条")
            prev = point

# 初始化
os.makedirs(out_pre, exist_ok=True)
stage_data = [[] for _ in range(3)]

# 处理所有文件
for idx, file_path in enumerate(input_files, 1):
    process_file(file_path, idx)

# 写入输出文件
for stage_idx in range(3):
    output_path = output_files[stage_idx]
    with jsonlines.open(output_path, 'w') as writer:
        writer.write_all(stage_data[stage_idx])
    print(f"\nStage {stage_idx+1}:")
    print(f"总数据量: {len(stage_data[stage_idx])}")
    print(f"保存路径: {output_path}")
    
    # 更新数据集配置
    dataset_key = f"{data_name}_{model_name}_stage_{stage_idx+1}_tem_{tem}"
    new_entry = {
        dataset_key: {
            "file_name": os.path.relpath(output_path, '/data/qq/LLaMA-Factory/data'),
            "columns": {"prompt": "question", "response": "solution"}
        }
    }
    update_data_info(new_entry)

print("\n数据处理全部完成！各阶段统计：")
for i, data in enumerate(stage_data, 1):
    print(f"Stage {i}: {len(data)} 条")