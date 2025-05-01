import json
import matplotlib.pyplot as plt
import os
import matplotlib.ticker as ticker

def read_jsonl_file(file_path):
    """
    读取单个 .jsonl 文件，返回该文件的日志列表（每行对应一个日志字典）。
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:  # 跳过空行
                data.append(json.loads(line))
    return data

def combine_logs_in_order(file_list):
    """
    按给定的 file_list 顺序依次读取并合并日志，不进行 current_steps 排序，
    同时记录每个文件结束时在合并列表中的位置作为交界点。
    返回合并后的日志列表和交界点列表（不包括最后一个文件的末尾）。
    """
    all_logs = []
    boundaries = []  # 记录每个文件合并后的结束位置（基于1计数）
    for file_path in file_list:
        logs = read_jsonl_file(file_path)
        logs = [log for log in logs if 'loss' in log]
        all_logs.extend(logs)
        boundaries.append(len(all_logs))
    # 最后一个边界是整个日志末尾，不作为交界标识
    return all_logs, boundaries[:-1]

def reindex_current_steps(log_data):
    """
    将日志的 current_steps 重新编号，从 1 开始连续到总长度。
    """
    for i, entry in enumerate(log_data, start=1):
        entry["current_steps"] = i
    return log_data

def plot_and_save_loss(log_data, boundaries, save_path):
    """
    根据日志中的 current_steps 和 loss 绘制曲线，并保存到指定路径。
    """
    log_data = [log for log in log_data if 'loss' in log]
    steps = [log['current_steps'] for log in log_data]
    losses = [log['loss'] for log in log_data]
    
    plt.figure(figsize=(16, 6))
    plt.plot(steps, losses, marker='None', linestyle='-', color='#1f77b4')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.title('Loss over Training Steps on Stage')
    plt.grid(True)

    ax = plt.gca()
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    # 在每个文件交界处（记录边界位置对应点）添加红色空心星形符号
    for boundary in boundaries:
        # boundary 基于1计数，列表索引为 boundary-1
        if boundary - 1 < len(log_data) and 'loss' in log_data[boundary - 1]:
            x_val = log_data[boundary - 1]['current_steps']
            y_val = log_data[boundary - 1]['loss']
            plt.plot(x_val, y_val, marker='o', markersize=2,
                     markerfacecolor='orange', markeredgecolor='orange', linestyle='None',
                     label='File Boundary' if boundary == boundaries[0] else None)
            
    # 如果目录不存在则创建目录
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.savefig(save_path)
    print(f"图像已保存到：{save_path}")
    plt.show()

def main():
    # 1. 指定你要读取的日志文件顺序
    file_list = [
        # '/data/qq/LLaMA-Factory/saves/Qwen2.5-Math-7B/lora/math-merge/base/5.0e-05/trainer_log.jsonl'
        '/data/qq/LLaMA-Factory/saves/Qwen2.5-Math-7B/lora/math-merge/stage_1_tem_0.7/5.0e-05/trainer_log.jsonl',
        '/data/qq/LLaMA-Factory/saves/Qwen2.5-Math-7B/lora/math-merge/stage_2_tem_0.7/5.0e-05/trainer_log.jsonl',
        '/data/qq/LLaMA-Factory/saves/Qwen2.5-Math-7B/lora/math-merge/stage_3_tem_0.7/5.0e-05/trainer_log.jsonl'
        # ... 继续添加 ...
    ]
    # 2. 按指定顺序依次读取并合并日志
    all_logs, boundaries = combine_logs_in_order(file_list)
    
    # 3. 重新编号 current_steps
    all_logs = reindex_current_steps(all_logs)
    
    save_path = '/data/qq/models/Qwen/Qwen2.5-Math-7B/math_eval/math-merge/loss_merge/stage_loss_merge.png'
    
    # 4. 绘制并保存 loss 曲线
    plot_and_save_loss(all_logs, boundaries, save_path)

if __name__ == '__main__':
    main()
