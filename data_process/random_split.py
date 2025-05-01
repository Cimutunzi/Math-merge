import json

input_file = '/data/qq/models/Qwen/Qwen2.5-Math-7B/math_eval/math-merge/Recording/train_direct_seed0_t0.7_n_sample_16.jsonl'  # 输入文件路径
out_pre = f'/data/qq/models/Qwen/Qwen2.5-Math-7B/math_eval/math-merge/data/tem_0.7'

output_file1 = f'{out_pre}/random/level_1.jsonl'  # 输出文件路径
output_file2 = f'{out_pre}/random/level_2.jsonl'  # 输出文件路径
output_file3 = f'{out_pre}/random/level_3.jsonl'  # 输出文件路径

# 定义每部分的行数
split_sizes = [2285, 3434, 3275]

# 读取所有数据
with open(input_file, 'r', encoding='utf-8') as infile:
    data = []
    for line in infile:
        obj = json.loads(line)
        # 仅保留需要的字段
        filtered_obj = {
            'idx': obj.get('idx'),
            'question': obj.get('question'),
            'answer': obj.get('solution')
        }
        data.append(filtered_obj)


# 确保数据足够分割
assert sum(split_sizes) == len(data), f"数据行数不一致: {sum(split_sizes)} != {len(data)}"

# 将数据分割成三部分并保存到新的文件中
output_files = [output_file1, output_file2, output_file3]

start_idx = 0
for i, split_size in enumerate(split_sizes):
    end_idx = start_idx + split_size
    # 获取每部分的数据
    split_data = data[start_idx:end_idx]
    
    # 写入对应的文件
    with open(output_files[i], 'w', encoding='utf-8') as outfile:
        for item in split_data:
            outfile.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    # 更新开始索引
    start_idx = end_idx

print("文件已分割并保存为：", output_file1, output_file2, output_file3)
