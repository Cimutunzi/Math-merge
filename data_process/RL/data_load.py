import jsonlines
import random

def load_data(input_file, level_boundaries):
    level_1_data = []
    level_2_data = []
    level_3_data = []
    with jsonlines.open(input_file) as reader:
        for obj in reader:
            # 修复：添加try-except处理可能的键缺失
            try:
                score_array = obj['score']
                total_elements = len(score_array)
                true_count = score_array.count(True)
                true_percentage = (true_count / total_elements) * 100 if total_elements else 0
                
                # 根据准确率分级
                if level_boundaries["level_1"][0] <= true_percentage <= level_boundaries["level_1"][1]:
                    level_1_data.append({
                        'idx': obj['idx'],
                        'question': obj['question'],
                        'answer': obj.get('solution', ''),  # 使用get避免KeyError
                        'accuracy': true_percentage
                    })
                elif level_boundaries["level_2"][0] <= true_percentage < level_boundaries["level_2"][1]:
                    level_2_data.append({
                        'idx': obj['idx'],
                        'question': obj['question'],
                        'answer': obj.get('solution', ''),
                        'accuracy': true_percentage
                    })
                elif level_boundaries["level_3"][0] <= true_percentage < level_boundaries["level_3"][1]:
                    level_3_data.append({
                        'idx': obj['idx'],
                        'question': obj['question'],
                        'answer': obj.get('solution', ''),
                        'accuracy': true_percentage
                    })
            except KeyError as e:
                print(f"Missing key in object: {e}")
                continue
    return [level_1_data, level_2_data, level_3_data]

def get_ratio_index(epoch):
    """epoch映射逻辑（示例：奇数epoch用第一个比例，偶数epoch轮换）"""
    if epoch in {1}:  # 自定义多对一映射
        return 0
    elif epoch in range(2,4):
        return 1
    else:
        return 2      

def sample_data(data, ratio):
    """根据比例抽样数据"""
    sampled = []
    for i, dataset in enumerate(data):
        if ratio[f"file_{i+1}"] <= 0:
            continue
        
        # 计算需要抽样的数量
        n_samples = int(len(dataset) * ratio[f"file_{i+1}"])
        if n_samples == 0:
            continue
        
        # 随机抽样（带种子保证可复现）
        sampled.extend(random.sample(dataset, min(n_samples, len(dataset))))
    return sampled

def main(input_file, epoch):
    # 配置参数
    level_boundaries = {
        "level_1": (60, 100),
        "level_2": (20, 60),
        "level_3": (0, 20)
    }
    
    sample_ratios = [
        {"file_1": 1, "file_2": 0, "file_3": 0},    # 模式0
        {"file_1": 0.1, "file_2": 1, "file_3": 0},  # 模式1
        {"file_1": 0.1, "file_2": 0.1, "file_3": 1} # 模式2
    ]
    
    # 获取当前epoch对应的数据比例
    ratio_index = get_ratio_index(epoch)
    current_ratio = sample_ratios[ratio_index]
    
    # 加载并分类数据
    classified_data = load_data(input_file, level_boundaries)
    
    # 抽样合并数据
    final_dataset = sample_data(classified_data, current_ratio)
    
    # 返回结构化结果
    return {
        "epoch": epoch,
        "ratio_index": ratio_index,
        "dataset_size": len(final_dataset),
        "samples": final_dataset
    }

if __name__ == "__main__":
    # 使用示例
    input_path = "/data/qq/models/Qwen/Qwen2.5-Math-7B/math_eval/math-merge/Recording/train_direct_seed0_t0.7_n_sample_16.jsonl"
    
    # 测试不同epoch
    for e in [1, 2, 3, 4, 5]:
        result = main(input_path, e)
        print(f"Epoch {e}: Using ratio {result['ratio_index']}")
        print(f"Total samples: {result['dataset_size']}")
        print("--"*30)