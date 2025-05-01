import argparse
import pandas as pd
import json5   # 宽容解析 JSON 字符串
import json
import re
from tqdm import tqdm
from volcenginesdkarkruntime import Ark
import random

def retry_failed(input_file, output_file, example_1, example_2):
    

    client = Ark(base_url="https://ark.cn-beijing.volces.com/api/v3")
    f_file = '/data/qq/data_process/example/fail.jsonl'
    failed_index_file = '/data/qq/data_process/example/failed_idx.log'
    # 加载所有行
    with open(input_file, "r", encoding="utf-8") as f:
        all_lines = f.readlines()

    # 加载失败的索引
    with open(failed_index_file, "r") as f:
        failed_indices = [int(line.strip()) for line in f if line.strip().isdigit()]

    # 打开输出文件
    with open(output_file, "w", encoding="utf-8") as jsonl_file, \
        open(f_file, "w", encoding="utf-8") as fail_file:
        for idx in tqdm(failed_indices, desc="Retrying Failed Indices"):
            try:
                line = all_lines[idx]
                data = json.loads(line)
                problem = data.get("problem", "")
                solution = data.get("solution", "")
                user_content = f"problem:{problem}\nsolution:{solution}"

                completion = client.chat.completions.create(
                    model="ep-20241221153657-69vts",
                    messages=[
                        {
                            "role": "system",
                            "content": f"""你是豆包，是由字节跳动开发的 AI 助手。
                                        我会给你一个数学题和它的答案，请你将答案拆解为多个步骤，每一步之间用 <step-devide> 分隔。

                                        请将拆解后的结果放入一个 JSON 对象中，**只返回如下格式的 JSON**：
                                        {{ "solution": "step1<step-devide>step2<step-devide>..." }}

                                        注意：
                                        - 必须是合法 JSON 格式，键名用英文双引号包裹。
                                        - 不要输出 markdown、解释、代码块标记（如 ```）或题干原文。
                                        - 数学表达式（如 $i^5$、\\frac{{1}}{{i}} 等）应作为字符串包在 solution 字段中返回。
                                        - 所有输出必须作为一个单行字符串包裹在 "solution" 中，不能换行。
                                        - 特别注意！只能在原本的solution中插入<step-devide>，不要修改其他任何词句和符号。

                                        示例：
                                        原数据:
                                        {example_1}
                                        输出:
                                        {example_2}
                                        """
                        },
                        {"role": "user", "content": f"待处理的数据:\n{user_content}"},
                    ],
                )

                print("=== 模型输出 ===")
                print(completion.choices[0].message.content)

                json_data = json5.loads(completion.choices[0].message.content)
                stepwise_solution = json_data.get("solution", "").strip()
                data["solution"] = stepwise_solution
                jsonl_file.write(json.dumps(data, ensure_ascii=False) + "\n")
            except:
                # print(f"解析错误,在字符串: {json_str}")
                print(f"最终失败，跳过 idx = {data['idx']}")
                fail_file.write(json.dumps(data, ensure_ascii=False) + "\n")
                # print(f"输入内容: {user_content}")
                continue


    print(f"处理完成，JSONL 文件已保存到 {output_file}")

if __name__ == "__main__":
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description="步骤划分")
    parser.add_argument("--input", required=True, help="输入的 Jsonl 文件路径")
    parser.add_argument("--output", required=True, help="输出的 JSONL 文件路径")
    # parser.add_argument("--failed_file", required=True, help="错误数据idx")
    parser.add_argument("--example_1", required=True, help="示例 JSON 格式数据，用于系统消息")
    parser.add_argument("--example_2", required=True, help="示例 JSON 格式数据，用于系统消息")

    # 解析参数
    args = parser.parse_args()

    # 调用主函数
    retry_failed(args.input, args.output, args.example_1, args.example_2)