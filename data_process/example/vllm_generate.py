import json
from vllm import LLM, SamplingParams
from tqdm import tqdm
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
# 初始化模型
model = LLM(
            model="/data/qq/models/deepseek-math-7b-instruct",
            tensor_parallel_size=4,  
            gpu_memory_utilization=0.9,
            dtype="float16" 
)  # 例如 "meta-llama/Llama-2-7b-chat-hf"

# 设置采样参数（可调整）
sampling_params = SamplingParams(
    temperature=0.9,
    top_p=0.9,
    max_tokens=2048,
    frequency_penalty=0.2,
    stop_token_ids=['100001']
    # stop = ['•']
)

# 输入输出路径
input_path = "/data/qq/data_process/example/end_solution_1-2.jsonl"
output_path = "/data/qq/data_process/example/deepseek_ins_start_solution_1-2.jsonl"
fail_idx = [1124,1591,3827,4038,4044,4230,5975,6739,8575,9107]
# 收集所有 prompts

# def build_prompt(question: str, solution: str) -> str:
#     return (
#     "You are given a math problem and its final answer.\n"
#     "Think step by step and write down your reasoning process clearly before arriving at the final answer.\n\n"
#     "### Problem:\n"
#     f"{question}\n"
#     "### Final Answer:\n"
#     f"{solution}\n"
#     "### Reasoning (Chain of Thought):\n"
# )

# def build_prompt(question: str, solution: str) -> str:
#     return (
#     'Below is a math question and the second half of its solution.'
#     'Please generate the first half of the solution that logically connects to the given second part.'
#     '### Problem:'
#     f'{question}'
#     '### Solution (ending part):'
#     f'{solution}'
#     '### Solution (starting part):'
# )
# def build_prompt(question: str, solution: str) -> str:
#     return (
#         'Below is a math problem along with the second half of its solution.\n '
#         'Your task is to generate the first part of the solution, starting from the problem statement and stopping right before the seconde part of the solution.\n '
#         'Remember that you only need to give the first part of the answer, not the entire answer.\n '
#         '### Example Problem:\n'
#         'Consider the given functions: $$\\begin{array}{ccc}\nf(x) & = & 5x^2 - \\frac{1}{x}+ 3\\\\\ng(x) & = & x^2-k\n\\end{array}$$If $f(2) - g(2) = 2$, what is the value of $k$?\n\nWe substitute ' 
#         '$f(2) = 5(2)^2 - \\frac{1}{2} + 3 = \\frac{45}{2}$ and $g(2) = (2)^2 - k = 4 - k$. So $f(2) - g(2) = 2$ gives us $\\frac{45}{2} - 4 + k=2$.\n'
#         '### Example Solution (second part):\n'
#         'Solving for $k$, we find $k = \\frac{4}{2} - \\frac{45}{2} + \\frac{8}{2}$ so $\\boxed{k = \\frac{-33}{2}}$.\n'
#         '### Example Solution (first part):\n'
#         'We substitute $f(2) = 5(2)^2 - \\frac{1}{2} + 3 = \\frac{45}{2}$ and $g(2) = (2)^2 - k = 4 - k$. So $f(2) - g(2) = 2$ gives us $\\frac{45}{2} - 4 + k=2$.\n'
#         '### Problem:\n'
#         f'{question}\n'
#         '### Solution (second part):\n'
#         f'{solution}\n'
#         '### Solution (first part):\n'
#     )
# def build_prompt(question: str, solution: str) -> str:
#     return (
#         'User:\n'
#         'Problem:\n'
#         f'{question}\n'
#         'Solution (second part):\n'
#         f'{solution}\n'
#         'Please only generate the first part of the solution, '
#         'starting from the problem statement and stopping right before the seconde part of the solution.\n\n'
#         'Assistant:'
#     )
def build_prompt(question: str, solution_end: str) -> str:
    return (
        "你是一个数学解题专家。请仔细执行以下步骤：\n"
        "1. 仔细阅读给出的数学问题\n"
        "2. 分析已提供的解答后半部分内容\n"
        "3. 生成一个精确匹配的前半部分解答，要求：\n"
        "   - 必须与后半部分形成完整的逻辑流\n"
        "   - 必须使用相同的数学符号和术语\n"
        "   - 绝对不要重复后半部分已有的任何内容\n"
        "   - 以自然过渡到给定后半部分的方式结束\n\n"
        "### 题目：\n"
        f"{question}\n\n"
        "### 已知的解答后半部分：\n"
        f"{solution_end}\n\n"
        "### 需要生成的前半部分解答（确保只生成前半部分，且不要包含任何后续内容）：\n"
        "首先，"
    )

# 读取数据并构建 prompts
prompts = []
raw_data = []
num = 0
with open(input_path, "r",encoding="utf-8") as infile:
    for line in infile:
        item = json.loads(line)
        # if item['idx'] in fail_idx:
            
        question = item["problem"]  # 假设字段名为 "problem"
        answer_end = item["answer_end"]  # 假设字段名为 "answer_end"
        prompt = build_prompt(question, answer_end)
        if num % 1000 == 0:
            print('='*100)
            print(prompt)
        prompts.append(prompt)
        raw_data.append(item)
        num += 1
            # if num >5: break


# 使用模型生成
outputs = model.generate(prompts, sampling_params)

# 写入带有生成结果的新 JSONL 文件
with open(output_path, "w", encoding="utf-8") as outfile:
    for item, output in tqdm(zip(raw_data, outputs)):
        item["answer_start"] = output.outputs[0].text.strip()  # 添加生成结果
        json.dump(item, outfile, ensure_ascii=False)
        outfile.write("\n")
        if not output.outputs[0].text.strip():
            print(f'生成为空，idx为{item["idx"]}')

            


print(f"生成完毕，共处理 {len(prompts)} 条，结果保存在：{output_path}")
