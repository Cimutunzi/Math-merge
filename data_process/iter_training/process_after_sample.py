import jsonlines
import os
import json
import sys

iteration = 1
# 设置文件路径
# input_file = f'/data/qq/models/Qwen/Qwen2.5-Math-7B/math_eval/iter/math-merge/Recording/iter_{iteration-1}_qwen_box_seed0_t0.7_n_sample_4.jsonl'  
# 输入文件路径
out_pre = f'/data/qq/math-evaluation-harness/data/math-merge'
train_out_pre = f'/data/qq/LLaMA-Factory/data/Qwen2.5-Math-7B/math-merge/iter'

output_file_1 = f'{train_out_pre}/iter_{iteration}.jsonl'  # 输出文件路径
output_file_2 = f'{out_pre}/iter_{iteration}.jsonl'  # 输出文件路径
train_data_file = f'{train_out_pre}/iter_{iteration-1}.jsonl'
model_name = 'Qwen2.5-Math-7B'
tem = '0.7'
data_name = 'math-merge'

# 定义准确率区间
level_boundaries = {
    "sample": (0, 25),
    "train": (25, 100)
}

# 用于存储符合条件的数据
sample_data = []
train_data = []

# 读取数据并处理
# 读取现有训练数据（避免重复数据）
def load_existing_idx():
    existing_idx = set()  # 用于存储已存在的 idx
    for i in range(1, iteration):  # 从 1 到 iteration-1，动态加载之前所有的迭代数据
        file_path = f'{train_out_pre}/iter_{i}.jsonl'
        if os.path.exists(file_path):
            with jsonlines.open(file_path, 'r') as reader:
                for obj in reader:
                    existing_idx.add(obj['idx'])  # 只保存 idx
    print(f"Existing idx: {len(existing_idx)} items loaded.")
    return existing_idx

def load_existing_data(file_path):
    data = []  # 使用 set 来去重
    if os.path.exists(file_path):
        with jsonlines.open(file_path, 'r') as reader:
            for obj in reader:
                data.append(obj)
                # data.append(json.dumps(obj))  # 将每条记录转为 JSON 字符串存储
    print(data[:1])
    return data


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
 
    print(f"数据集 {data_name}_{model_name}_iter_{iteration} 信息已成功更新到 data_info.json 文件")

def update_train_count_and_filter(train_data, iterations=3):
    updated_data = train_data.copy()
    for i in range(3):
        # 增加每条数据的训练次数
        for item in updated_data:
            item['train_count'] += 1
        updated_data = [item for item in updated_data if item['train_count'] < 4]
        iteration_now = iteration + i
        output_file =f'{train_out_pre}/iter_{iteration_now}.jsonl' 
        with jsonlines.open(output_file, mode='w') as writer:
            for data in updated_data:
                writer.write(data)
        new_data = {
            f"{data_name}_{model_name}_iter_{iteration_now}": {
                "file_name": os.path.relpath(output_file, '/data/qq/LLaMA-Factory/data'),
                "columns": {
                    "prompt": "question",
                    "response": "answer"
                    }
                }
            }
        update_data_info(new_data)
        print(f"Iteration complete, {len(updated_data)} items remaining.")

    print("条件满足，退出程序。")
    sys.exit()


# def generate_yaml():
# 加载现有训练数据
existing_train_data = load_existing_data(train_data_file)

if iteration > 6:
    with jsonlines.open(f'{out_pre}/iter_6.jsonl') as reader:
        for obj in reader:
            obj['train_count'] = 0
            existing_train_data.append({ 'idx': obj['idx'], 'question': obj['question'], 'answer': obj['solution'],'train_count': obj['train_count']})
    update_train_count_and_filter(existing_train_data)
# 读取新数据并分类
with jsonlines.open(input_file) as reader:
    existing_idx = load_existing_idx()
    for obj in reader:
        score_array = obj.get('score', [])[:4]  # 取 score 数组的前四个元素
        total_elements = len(score_array)  # 数组长度
        true_count = score_array.count(True)  # True 的数量
        # 计算准确率（True 的比例）
        acc = (true_count / total_elements) * 100 if total_elements else 0
        num = 0
        
        if acc >= level_boundaries["train"][0] and acc <= level_boundaries["train"][1]:
            # 训练数据区间：25% - 100%
            if obj['idx'] not in existing_idx:
                obj['train_count'] = 0  # 新数据的train_count设为0，表示第一次加入
                existing_train_data.append({ 'idx': obj['idx'], 'question': obj['question'], 'answer': obj['solution'], 'accuracy': acc, 'train_count': obj['train_count']})

        elif acc >= level_boundaries["sample"][0] and acc < level_boundaries["sample"][1]:
            # 采样数据区间：0% - 25%
            sample_data.append(obj)

train_data = [item for item in existing_train_data if item['train_count'] < 3]
# 更新训练次数：每次训练时增加
for data in train_data:
    data['train_count'] += 1
# 确保输出目录存在
os.makedirs(out_pre, exist_ok=True)

# 将训练数据和采样数据保存到文件
with jsonlines.open(output_file_1, mode='w') as writer:
    for data in train_data:
        writer.write(data)

with jsonlines.open(output_file_2, mode='w') as writer:
    for data in sample_data:
        writer.write(data)

new_data = {
        f"{data_name}_{model_name}_iter_{iteration}": {
            "file_name": os.path.relpath(output_file_1, '/data/qq/LLaMA-Factory/data'),
            "columns": {
                "prompt": "question",
                "response": "answer"
            }
        }
    }

update_data_info(new_data)

print(f"数据处理完成，已将 {len(train_data)} 条训练数据写入 {output_file_1}")
print(f"数据处理完成，已将 {len(sample_data)} 条采样数据写入 {output_file_2}")


