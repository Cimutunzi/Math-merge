import json
import random
import jsonlines

def init_data(ori_file, file_path)：
    with jsonlines.open(ori_file, 'r') as reader,
        with jsonlines.open(file_path, 'w') as writer:
            for obj in reader:
                writer.write(obj)

# 读取数据集
def load_data(file_path):
    data = []
    with jsonlines.open(file_path, 'r') as reader:
        for obj in reader:
            data.append(obj)
    return data

# 保存更新后的 jsonl 数据
def save_data(file_path, data):
    with jsonlines.open(file_path, 'w') as writer:
        for obj in data:
            writer.write(obj)

# 选择需要训练的数据
def filter_data_for_training(data):
    # 选择训练次数小于3的样本
    return [item for item in data if item['train_count'] < 3]

# 更新训练次数
def update_train_count(data, trained_data):
    for item in trained_data:
        for d in data:
            if d['id'] == item['id']:
                d['train_count'] += 1
                break
    return data


# 迭代训练
def iterative_training(data_file, new_data_file, max_epochs):
    # 加载现有的数据集
    data = load_data(data_file)
    generate_data()
    generate_yaml()
    for epoch in range(max_epochs):
        print(f"Epoch {epoch+1}/{max_epochs}")

        # 加载新的数据
        new_data = load_data(new_data_file)

        # 将新数据加入到当前数据集中
        data.extend(new_data)

        # 过滤出可用的数据（训练次数小于3的样本）
        data_to_train = filter_data_for_training(data)

        # 随机选择一批数据进行训练
        batch_size = min(32, len(data_to_train))
        batch_data = random.sample(data_to_train, batch_size)

        # 进行训练
        trained_data = train_on_batch(batch_data)

        # 更新训练次数
        data = update_train_count(data, trained_data)

        # 剔除训练次数已达到3次的样本
        data = [d for d in data if d['train_count'] < 3]

        # 保存更新后的数据集
        save_data(data_file, data)

        print(f"Epoch {epoch+1} training completed.")

# 假设数据文件路径如下
data_file = 'dataset.json'
new_data_file = 'new_data.json'

# 运行迭代训练，设定最大迭代次数
iterative_training(data_file, new_data_file, max_epochs=10)
