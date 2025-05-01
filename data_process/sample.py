import json
import random
import os

# 读取原始 JSON 数据
data = []
with open('/data/qq/data/ScaleQuest-Math/train.json', 'r', encoding='utf-8') as file:
    for line in file:
        line = json.loads(line.strip())
        data.append({'problem': line['query'], 'solution': line['response']})

# 随机抽样1万条数据
sampled_data = random.sample(data, 20000)
output_dir = '/data/qq/math-evaluation-harness/data/ScaleQuest-Math/'
os.makedirs(output_dir, exist_ok=True) 
# 保存抽样后的数据到新文件
with open('/data/qq/math-evaluation-harness/data/ScaleQuest-Math/train.jsonl', 'w', encoding='utf-8') as file:
    for entry in sampled_data:
        json.dump(entry, file, ensure_ascii=False)
        file.write('\n')