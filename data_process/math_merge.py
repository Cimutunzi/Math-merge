import json

# 文件路径
file1_path = "/data/qq/LLaMA-Factory/data/math/train.jsonl"
file2_path = "/data/qq/LLaMA-Factory/data/math/test.jsonl"
output_path = "/data/qq/math-evaluation-harness/data/math/raw_hard_v1.jsonl"

# 剔除 file1 中 Level 1 和 Level 2 的数据
filtered_data = []
with open(file1_path, "r", encoding="utf-8") as f1:
    for line in f1:
        data = json.loads(line.strip())
        if data["level"] in ["Level 3","Level 4", "Level 5"]:  # 保留 Level 3 及以上的数据
            filtered_data.append(data)

# 读取 file2 的数据
# with open(file2_path, "r", encoding="utf-8") as f2:
#     for line in f2:
#         data = json.loads(line.strip())
#         filtered_data.append(data)  # 直接合并 file2 的所有数据
with open(file2_path, "r", encoding="utf-8") as f2:
    for line in f2:
        data = json.loads(line.strip())
        if data["level"] in ["Level 3","Level 4", "Level 5"]:  # 保留 Level 3 及以上的数据
            filtered_data.append(data)
# 将合并后的数据写入新文件
with open(output_path, "w", encoding="utf-8") as outfile:
    for data in filtered_data:
        outfile.write(json.dumps(data, ensure_ascii=False) + "\n")

print(f"合并完成，结果已保存到 {output_path}")