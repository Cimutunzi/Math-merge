import json

# 定义文件路径
stage1_file = '/data/qq/data/math-merge/data/Qwen2.5-Math-1.5B/stage_1.jsonl'
stage2_file = '/data/qq/data/math-merge/data/Qwen2.5-Math-1.5B/stage_2.jsonl'
level1_file = '/data/qq/models/Qwen/Qwen2.5-Math-1.5B/math_eval/math-merge/data/tem_0.7/level_1.jsonl'
level2_file = '/data/qq/models/Qwen/Qwen2.5-Math-1.5B/math_eval/math-merge/data/tem_0.7/level_2.jsonl'

# 创建一个集合存储level1的idx
level1_idx_set = set()

# 读取level1文件并获得所有的idx
with open(level1_file, 'r') as file:
    for line in file:
        data = json.loads(line.strip())
        level1_idx_set.add(data['idx'])

# 创建一个集合存储level2的idx
level2_idx_set = set()

# 读取level2文件并获得所有的idx
with open(level2_file, 'r') as file:
    for line in file:
        data = json.loads(line.strip())
        level2_idx_set.add(data['idx'])

# 打开输出文件
output_stage1_level1 = open('/data/qq/data/math-merge/data/Qwen2.5-Math-1.5B/level_1.jsonl', 'w')
output_stage2_level2 = open('/data/qq/data/math-merge/data/Qwen2.5-Math-1.5B/level_2.jsonl', 'w')

# 处理stage1文件
with open(stage1_file, 'r') as file:
    for line in file:
        data = json.loads(line.strip())
        if data['idx'] in level1_idx_set:
            output_stage1_level1.write(json.dumps(data) + '\n')

# 处理stage2文件
with open(stage2_file, 'r') as file:
    for line in file:
        data = json.loads(line.strip())
        if data['idx'] in level2_idx_set:
            output_stage2_level2.write(json.dumps(data) + '\n')

# 关闭输出文件
output_stage1_level1.close()
output_stage2_level2.close()
