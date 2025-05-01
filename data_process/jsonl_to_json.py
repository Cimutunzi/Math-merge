import json
input_file = '/data/qq/LLaMA-Factory/data/gsm8k/gsm8k_train.jsonl'
out_file = '/data/qq/LLaMA-Factory/data/gsm8k_train.json'
# 读取 JSONL 文件
with open(input_file, "r") as file:
    lines = file.readlines()

# 转换为 JSON 数组
json_data = [json.loads(line) for line in lines]

# 写入 JSON 文件
with open(out_file, "w") as file:
    json.dump(json_data, file, indent=4)

print("转换完成！")