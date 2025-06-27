import json

# 读取第一个JSONL文件
with open("/data/qq/data/math-merge/data/Qwen2.5-Math-1.5B/ori_level/stage_3.jsonl", "r") as file1:
    data1 = [json.loads(line) for line in file1]

# 读取第二个JSONL文件
with open("/data/qq/data/math-merge/data/Qwen2.5-Math-1.5B/ori_level/stage_3-hint.jsonl", "r") as file2:
    data2 = [json.loads(line) for line in file2]

# 筛选出level为"Level 4"的数据
filtered_data = [entry for entry in data1 if entry["level"] == "Level 4"]

# 合并两个数据集
merged_data = filtered_data + data2

# 打乱合并后的数据
import random
random.shuffle(merged_data)

# 保存合并后的数据到指定文件
output_path = "/data/qq/data/math-merge/data/Qwen2.5-Math-1.5B/ori_level/stage_3_final.jsonl"  # 指定保存的文件路径
with open(output_path, "w") as output_file:
    for entry in merged_data:
        json.dump(entry, output_file)
        output_file.write("\n")

print(f"合并后的数据已保存到: {output_path}")
