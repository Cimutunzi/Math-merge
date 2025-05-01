import json
import matplotlib.pyplot as plt
import os
import matplotlib.ticker as ticker
import numpy as np

def read_jsonl_file(file_path):
    """ 读取单个 .jsonl 文件，返回日志列表。 """
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

def combine_logs_in_order(file_list):
    """ 按顺序读取并合并日志，同时记录文件的边界点。 """
    all_logs = []
    boundaries = []
    for file_path in file_list:
        logs = read_jsonl_file(file_path)
        logs = [log for log in logs if 'loss' in log]
        all_logs.extend(logs)
        boundaries.append(len(all_logs))  # 记录每个文件结束的位置
    return all_logs, boundaries[:-1]  # 最后一个边界不需要

def reindex_current_steps(log_data):
    """ 重新编号 current_steps，使其从 1 开始连续编号。 """
    for i, entry in enumerate(log_data, start=1):
        entry["current_steps"] = i
    return log_data

def smooth_curve(data, weight=0.9):
    """ 通过指数加权移动平均（EMA）平滑曲线。 """
    smoothed = []
    last = data[0]
    for point in data:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

def downsample_data(steps, losses, n=10):
    """ 降采样：每 n 个点取 1 个，同时保留边界点。 """
    return steps[::n], losses[::n]

def plot_and_save_loss(log_data, boundaries, save_path):
    """ 绘制平滑的 loss 曲线，并在边界处添加虚线标记。 """
    log_data = [log for log in log_data if 'loss' in log]
    steps = [log['current_steps'] for log in log_data]
    losses = [log['loss'] for log in log_data]

    # 平滑 loss
    smooth_losses = smooth_curve(losses, weight=0.9)

    # 降采样
    sampled_steps, sampled_losses = downsample_data(steps, smooth_losses, n=50)

    plt.figure(figsize=(16, 6))
    plt.plot(sampled_steps, sampled_losses, linestyle='-', color='orange')  # 绘制平滑曲线
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.title('Loss over Training Steps')
    plt.grid(True)

    ax = plt.gca()
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    # 在每个阶段的边界处添加虚线
    for boundary in boundaries:
        x_val = log_data[boundary - 1]['current_steps']
        plt.axvline(x=x_val, linestyle='--', color='black', alpha=0.8, linewidth=1.2, label='Stage Boundary' if boundary == boundaries[0] else None)

    # 确保目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.savefig(save_path)
    print(f"图像已保存到：{save_path}")
    plt.show()

def main():
    file_list = [
        # '/data/qq/LLaMA-Factory/saves/Qwen2.5-Math-7B/lora/math-merge/single/stage_1_tem_0.7/5.0e-05/trainer_log.jsonl',
        # '/data/qq/LLaMA-Factory/saves/Qwen2.5-Math-7B/lora/math-merge/single/stage_2_tem_0.7/5.0e-05/trainer_log.jsonl',
        # '/data/qq/LLaMA-Factory/saves/Qwen2.5-Math-7B/lora/math-merge/single/stage_3_tem_0.7/5.0e-05/trainer_log.jsonl'
        '/data/qq/LLaMA-Factory/saves/Qwen2.5-Math-7B/lora/math-merge/stage_1_tem_0.7/5.0e-05/trainer_log.jsonl',
        '/data/qq/LLaMA-Factory/saves/Qwen2.5-Math-7B/lora/math-merge/stage_2_tem_0.7/5.0e-05/trainer_log.jsonl',
        '/data/qq/LLaMA-Factory/saves/Qwen2.5-Math-7B/lora/math-merge/stage_3_tem_0.7/5.0e-05/trainer_log.jsonl'
    ]

    all_logs, boundaries = combine_logs_in_order(file_list)
    all_logs = reindex_current_steps(all_logs)
    
    save_path = '/data/qq/models/Qwen/Qwen2.5-Math-7B/math_eval/math-merge/loss_merge/stage_loss_merge.png'
    plot_and_save_loss(all_logs, boundaries, save_path)

if __name__ == '__main__':
    main()
