import json
import matplotlib.pyplot as plt
import os
import matplotlib.ticker as ticker

def read_jsonl_file(file_path):
    """读取 .jsonl 文件，返回日志列表。"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                log = json.loads(line)
                if 'loss' in log:  # 只保留包含 loss 的数据
                    data.append(log)
    return data

def smooth_curve(data, weight=0.9):
    """通过指数加权移动平均（EMA）平滑曲线。"""
    smoothed = []
    last = data[0]
    for point in data:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

def downsample_data(steps, losses, n=10):
    """降采样：每 n 个点取 1 个。"""
    return steps[::n], losses[::n]

def plot_loss_curves(log_data_1, log_data_2, save_path):
    """绘制两个 loss 曲线并保存。"""
    steps_1 = [log['current_steps'] for log in log_data_1]
    losses_1 = [log['loss'] for log in log_data_1]
    steps_2 = [log['current_steps'] for log in log_data_2]
    losses_2 = [log['loss'] for log in log_data_2]

    # 平滑 loss
    smooth_losses_1 = smooth_curve(losses_1, weight=0.9)
    smooth_losses_2 = smooth_curve(losses_2, weight=0.9)

    # 降采样
    sampled_steps_1, sampled_losses_1 = downsample_data(steps_1, smooth_losses_1, n=10)
    sampled_steps_2, sampled_losses_2 = downsample_data(steps_2, smooth_losses_2, n=10)

    plt.figure(figsize=(16, 6))

    # 绘制两条曲线
    plt.plot(sampled_steps_1, sampled_losses_1, linestyle='-', color='orange', label='diect_hard')  
    plt.plot(sampled_steps_2, sampled_losses_2, linestyle='-', color='blue', label='easy_to_hard')  

    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.title('Loss Comparison Between Two Models')
    plt.legend()  # 添加图例
    plt.grid(True)

    ax = plt.gca()
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    # 确保目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.savefig(save_path)
    print(f"图像已保存到：{save_path}")
    plt.show()

def main():
    file_1 = '/data/qq/LLaMA-Factory/saves/Qwen2.5-Math-7B/lora/math-merge/direct_hard/5.0e-05/trainer_log.jsonl'
    file_2 = '/data/qq/LLaMA-Factory/saves/Qwen2.5-Math-7B/lora/math-merge/easy_hard_2/5.0e-05/trainer_log.jsonl'

    log_data_1 = read_jsonl_file(file_1)
    log_data_2 = read_jsonl_file(file_2)

    save_path = '/data/qq/models/Qwen/Qwen2.5-Math-7B/math_eval/math-merge/loss_merge/loss_comparison.png'
    plot_loss_curves(log_data_1, log_data_2, save_path)

if __name__ == '__main__':
    main()