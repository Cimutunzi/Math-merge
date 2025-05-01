import json

input_file = "/data/qq/data_process/example/deepseek_ins_start_solution_1-2.jsonl"
output_file = "/data/qq/data_process/example/deepseek_ins_start_solution_1-2_merge.jsonl"
def escape_special_slashes(text):
    return text.replace("\b", "\\b").replace("\r", "\\r").replace("\f", "\\f")
with open(input_file, "r", encoding="utf-8") as infile, open(output_file, "w", encoding="utf-8") as outfile:
    for i, line in enumerate(infile):
        try:
            data = json.loads(line)
            # generated_start = escape_special_slashes(data.get("generated_answer_start", ""))
            generated_start = data.get("answer_start", "")
            # answer_end = escape_special_slashes(data.get("answer_end", ""))
            data["problem"] = data["problem"] + ' ' + generated_start
            # data["solution"] = answer_end
            outfile.write(json.dumps(data, ensure_ascii=False) + "\n")
        except json.JSONDecodeError as e:
            print(f"⚠️ 第{i+1}行 JSON 解析失败: {e}")
            print(f"内容片段: {line[:200]}...\n")
