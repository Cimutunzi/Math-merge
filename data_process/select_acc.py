import json

def filter_jsonl_by_accuracy(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:
        num = 0
        for line in infile:
            try:
                item = json.loads(line)
                # 检查 accuracy 字段，如果为 0 则跳过该项
                # if item.get('accuracy', 0) <= 43.75:
                if item.get('accuracy', 0) <= 67.5:
                    outfile.write(json.dumps(item, ensure_ascii=False) + '\n')
                    num+=1
            except json.JSONDecodeError:
                print(f"跳过无法解析的行：{line[:100]}")
            
    print(num)
# 使用方法
input_file = '/data/qq/LLaMA-Factory/data/Qwen2.5-Math-7B/math-merge/tem_0.7/stage_3-easy.jsonl'  # 输入文件路径
output_file = '/data/qq/data/math-merge/data/Qwen2.5-Math-7B/stage_3-only_level_2.jsonl'  # 输出文件路径

# input_file = '/data/qq/LLaMA-Factory/data/Qwen2.5-Math-1.5B/math-merge/tem_0.7/stage_3-easy.jsonl'  # 输入文件路径
# output_file = '/data/qq/data/math-merge/data/Qwen2.5-Math-1.5B/stage_3-only_level_2.jsonl'  # 输出文件路径

filter_jsonl_by_accuracy(input_file, output_file)