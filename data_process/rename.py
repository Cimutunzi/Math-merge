import json

# 输入和输出文件路径
input_file = '/data/qq/simpleRL-reason/eval/data/olympiadbench/test.jsonl'
output_file = '/data/qq/math-evaluation-harness/data/olympiadbench/test.jsonl'

# 字段名映射 (旧字段 -> 新字段)
field_mapping = {
    "final_answer": "solution",
    "solution": "cot",
    # 在这里添加更多的字段映射
}

# 处理文件
with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
    for line in infile:
        data = json.loads(line.strip())  # 读取每一行并解析为 JSON 对象
        data['final_answer'] = ' '.join(str(item) for item in data['final_answer'])
        # 重命名字段并将字段值转换为整数后转换为字符串
        # data = {field_mapping.get(key, key): str(int(value)) if isinstance(value, (int, float)) else str(value) for key, value in data.items()}
        data = {field_mapping.get(key, key): value for key, value in data.items()}  # 重命名并转换为字符串
        # 写入输出文件
        outfile.write(json.dumps(data, ensure_ascii=False) + '\n')

print(f"字段重命名完成，输出文件保存在：{output_file}")
