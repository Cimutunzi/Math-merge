import random
import os
import argparse
import time
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from datetime import datetime
from tqdm import tqdm
import jsonlines
import random
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from evaluate import evaluate
from utils import set_seed, load_jsonl, save_jsonl, construct_prompt
from parser import *
from trajectory import *
from data_loader import load_data
from python_executor import PythonExecutor
from model_utils import load_hf_lm_and_tokenizer, generate_completions
import json
import matplotlib.pyplot as plt
import seaborn as sns



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_names", default="gsm8k,math", type=str)
    parser.add_argument("--data_dir", default="./data", type=str)
    parser.add_argument("--model_name_or_path", default="gpt-4", type=str)
    parser.add_argument("--adpter_path", default="", type=str)
    parser.add_argument("--output_dir", default="./output", type=str)
    parser.add_argument("--prompt_type", default="tool-integrated", type=str)
    parser.add_argument("--split", default="test", type=str)
    parser.add_argument("--num_test_sample", default=-1, type=int) # -1 for full data
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--start", default=0, type=int)
    parser.add_argument("--end", default=-1, type=int)
    parser.add_argument("--temperature", default=0, type=float)
    parser.add_argument("--n_sampling", default=1, type=int)
    parser.add_argument("--top_p", default=1, type=float)
    parser.add_argument("--max_tokens_per_call", default=1024, type=int)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--use_vllm", action="store_true")
    parser.add_argument("--save_outputs", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--use_safetensors", action="store_true")
    parser.add_argument("--test_level", action="store_true")
    args = parser.parse_args()
    args.top_p = 1 if args.temperature == 0 else args.top_p # top_p must be 1 when using greedy sampling (vllm)
    return args

def acc_distribution(args, input_file):
    accuracies = []

    # 逐行读取 JSONL 文件
    with open(input_file, 'r') as f:
        for line in f:
            entry = json.loads(line.strip())  # 解析每一行 JSON 数据
            
            score_array = entry['score']
            total_elements = len(score_array)  # 数组长度
            true_count = score_array.count(True)  # True 的数量
            
            # 计算 true 占比
            true_percentage = (true_count / total_elements) * 100 if total_elements else 0
            accuracies.append(true_percentage)

    plt.figure(figsize=(10, 6))
    ax = sns.histplot(accuracies, bins=np.arange(0, 105, 10), kde=False, color='b', edgecolor='black')
    # 在每个柱子上方显示频数
    plt.xlim(0, 100)
    plt.xticks(np.arange(0, 105, 10))

    for p in ax.patches:
        height = p.get_height()
        width = p.get_width()
        x = p.get_x()
        ax.text(x + width / 2, height, f'{int(height)}', ha='center', va='bottom', fontsize=10)
        # 设置标题和标签
    plt.title('Accuracy Distribution (Histogram)', fontsize=16)
    plt.xlabel('Accuracy (%)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    # 保存图像到文件
    plt.tight_layout()  # 优化布局
    os.makedirs(f'{args.model_name_or_path}/math_eval/{args.data_names}/data/tem_{args.temperature}', exist_ok=True)
    plt.savefig(f'{args.model_name_or_path}/math_eval/{args.data_names}/data/tem_{args.temperature}/{args.split}_accuracy_distribution_{args.temperature}.png', dpi=300)  # 保存为 PNG 文件，分辨率为 300dpi
    # 显示图形（如果需要）
    plt.show()

def data_process(args, input_file):
    out_pre = f'{args.model_name_or_path}/math_eval/{args.data_names}/data/tem_{args.temperature}/'
    os.makedirs(out_pre, exist_ok=True)
    output_file_1 = f'{out_pre}/level_1.jsonl'  # 输出文件路径
    output_file_2 = f'{out_pre}/level_2.jsonl'  # 输出文件路径
    output_file_3 = f'{out_pre}/level_3.jsonl'  # 输出文件路径

    # 用于存储符合条件的数据
    filtered_data = []
    # 读取数据并处理
    with jsonlines.open(input_file) as reader:
        level_1_data = []
        level_2_data = []
        level_3_data = []
        num = 0
        # 遍历文件中的每一行数据
        for obj in reader:
            # 提取所需的字段
            if obj['test_level'] in [0]:
                level_1_data.append({ 'idx': obj['idx'], 'question': obj['question'], 'answer': obj['answer'], 'level': obj['test_level'] })
            if obj['test_level'] in [1,2,3]:
                level_2_data.append({ 'idx': obj['idx'], 'question': obj['question'], 'answer': obj['answer'], 'level': obj['test_level'] })
            if obj['test_level'] in [4]:
                level_3_data.append({ 'idx': obj['idx'], 'question': obj['question'], 'answer': obj['answer'], 'level': obj['test_level'] })

    # 将筛选后的数据写入新文件
    if not os.path.exists(out_pre):
        os.makedirs(out_pre)
        
    with jsonlines.open(output_file_1, mode='w') as writer:
        writer.write_all(level_1_data)

    with jsonlines.open(output_file_2, mode='w') as writer:
        writer.write_all(level_2_data)

    with jsonlines.open(output_file_3, mode='w') as writer:
        writer.write_all(level_3_data)

    print(f"数据处理完成，保存至 {output_file_1}")


def prepare_data(data_name, args):
    examples = load_data(data_name, args.split, args.data_dir)

    # sample `num_test_sample` from dataset
    if args.num_test_sample > 0:
        examples = random.sample(examples, args.num_test_sample)

    # shuffle
    if args.shuffle:
        random.shuffle(examples, seed=datetime.now().timestamp())

    # select start and end
    examples = examples[args.start:len(examples) if args.end == -1 else args.end]

    # get out_file name
    dt_string = datetime.now().strftime("%m-%d_%H-%M")
    model_name = "/".join(args.model_name_or_path.split("/")[-2:])
    adapter_split = args.adpter_path.split('/')[-1]
    out_file_prefix = f'{args.split}_{args.prompt_type}_seed{args.seed}_t{args.temperature}_n_sample_{args.n_sampling}'
    # out_file = f'{args.output_dir}/{model_name}/{data_name}/{out_file_prefix}_s{args.start}_e{args.end}_{dt_string}.jsonl'
    # os.makedirs(f'{args.output_dir}/{data_name}/', exist_ok=True)
    os.makedirs(f'{args.output_dir}/{data_name}/Recording/', exist_ok=True)
    out_file = f'{args.output_dir}/{data_name}/Recording/{out_file_prefix}.jsonl'
    # os.makedirs(f'{args.output_dir}/{data_name}', exist_ok=True)

    # load all processed samples
    processed_samples = []
    if not args.overwrite:
        processed_files = [f for f in os.listdir(f"{args.output_dir}/{data_name}/") if f.endswith(".jsonl") and f.startswith(out_file_prefix)]    
        for f in processed_files:
            processed_samples.extend(list(load_jsonl(f"{args.output_dir}/{data_name}/{f}")))

    # dedepulicate
    processed_samples = {sample['idx']: sample for sample in processed_samples}
    processed_idxs = list(processed_samples.keys())
    processed_samples = list(processed_samples.values())
    total_examples = len(examples)
    examples = [example for example in examples if example['idx'] not in processed_idxs]
    # print(f"Idx {args.start} - {args.end}: Remain {len(examples)}/{total_examples} samples.")
    return examples, processed_samples, out_file


def setup(args):
    # load model
    available_gpus = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
    if args.use_vllm:
        if args.adpter_path:
            
            llm = LLM(model=args.model_name_or_path, tensor_parallel_size=len(available_gpus), trust_remote_code=True,  enable_lora=True)
        else:
            llm = LLM(model=args.model_name_or_path, tensor_parallel_size=len(available_gpus), trust_remote_code=True)
        tokenizer = None
    else:
        llm, tokenizer =  load_hf_lm_and_tokenizer(
                model_name_or_path=args.model_name_or_path, 
                load_in_half=True,
                use_fast_tokenizer=True,
                use_safetensors=args.use_safetensors,
            )

    # infer & eval
    data_list = args.data_names.split(',')
    results = []
    for data_name in data_list:
        results.append(main(llm, tokenizer, data_name, args))
    
    # add "avg" result to data_list and results
    data_list.append("avg")
    results.append({
        "acc": sum([result["acc"] for result in results]) / len(results),
    })
    
    # print all results
    pad = max([len(data_name) for data_name in data_list])
    print("\t".join(data_name.ljust(pad, " ") for data_name in data_list))
    print("\t".join([f"{result['acc']:.1f}".ljust(pad, " ") for result in results]))


def main(llm, tokenizer, data_name, args):
    data_file = f"{args.data_dir}/{data_name}/{args.split}.jsonl"

    examples, processed_samples, out_file = prepare_data(data_name, args)
    print("=" * 50)
    print("data:", data_name, " ,remain samples:", len(examples))
    if len(examples) > 0:
        print(examples[0])

    # init python executor
    if "pal" in args.prompt_type:
        executor = PythonExecutor(get_answer_expr='solution()')
    else:
        executor = PythonExecutor(get_answer_from_stdout=True)

    samples = []
    for example in tqdm(examples, total=len(examples)):
        idx = example['idx']

        # parse question and answer
        example['question'] = parse_question(example, data_name)
        gt_cot, gt_ans = parse_ground_truth(example, data_name)
        full_prompt = construct_prompt(example, data_name, args)

        if idx == args.start:
            print(full_prompt)

        sample = {'idx': idx, 'question': example['question'], 'gt_cot': gt_cot, 'gt': gt_ans, 'prompt': full_prompt}

        # add remain fields
        for key in ['level', 'type', 'unit', 'solution_type', 'choices', 'solution', 'ques_type', \
            'ans_type', 'answer_type', 'dataset', 'subfield', 'filed', 'theorem', 'answer']:
            if key in example:
                sample[key] = example[key]
        samples.append(sample)


    # repeat n times
    input_prompts = [sample['prompt'] for sample in samples for _ in range(args.n_sampling)]
    remain_prompts = input_prompts
    remain_prompts = [(i, prompt) for i, prompt in enumerate(remain_prompts)]
    end_prompts = []

    max_func_call = 1 if args.prompt_type in ['cot', 'pal'] else 4

    # stop words TODO: make it more general
    stop_words = ["</s>"]

    if args.prompt_type in ['cot']:
        stop_words.extend(["\n\nQuestion:", "\n\nProblem:"])
    if args.prompt_type in ['pal', 'tool-integrated', 'tora']:
        stop_words.extend(["\n\n---", "```output"])
    elif args.prompt_type in ['wizard_zs', 'platypus_fs']:
        stop_words.extend(["Instruction", "Response"])
    print("Stop words:", stop_words)

    # start inference
    # measure time use
    start_time = time.time()
    for epoch in range(max_func_call):
        print("-" * 20, "Epoch", epoch)
        current_prompts = remain_prompts
        if len(current_prompts) == 0:
            break

        # get all outputs
        prompts = [item[1] for item in current_prompts]
        if args.use_vllm:
            if args.adpter_path:
                outputs = llm.generate(prompts, SamplingParams(
                                temperature=args.temperature,
                                top_p=args.top_p,
                                max_tokens=args.max_tokens_per_call,
                                n=1,
                                stop=stop_words),
                                lora_request=LoRARequest("adapter", 1, args.adpter_path)
                                )
            else:
                outputs = llm.generate(prompts, SamplingParams(
                                temperature=args.temperature,
                                top_p=args.top_p,
                                max_tokens=args.max_tokens_per_call,
                                n=1,
                                stop=stop_words,
                ))

            outputs = sorted(outputs, key=lambda x: int(x.request_id)) # sort outputs by request_id
            outputs = [output.outputs[0].text for output in outputs]
        else:
            print('DONT USE VLLM')
            if args.adpter_path:
                model = PeftModel.from_pretrained(
                llm,
                args.adpter_path,
                is_trainable=False
                )
                model = model.merge_and_unload()
            else:
                model = llm
            outputs = generate_completions(
                model=model,
                tokenizer=tokenizer,
                prompts=prompts,
                max_new_tokens=args.max_tokens_per_call,
                batch_size=16,
                stop_id_sequences=stop_words,
            )

        assert len(outputs) == len(current_prompts)

        # process all outputs
        remain_prompts = []
        remain_codes = []
        for (i, query), output in zip(current_prompts, outputs):
            output = output.rstrip()
            query += output
            if args.prompt_type == "pal":
                remain_prompts.append((i, query))
                if "```python" in output:
                    output = extract_program(query)
                remain_codes.append(output)
            elif args.prompt_type == "cot":
                end_prompts.append((i, query))
            elif ("boxed" not in output and output.endswith("```")):
                program = extract_program(query)
                remain_prompts.append((i, query))
                remain_codes.append(program)
            else:
                end_prompts.append((i, query))

        # execute the remain prompts
        remain_results = executor.batch_apply(remain_codes)
        for k in range(len(remain_prompts)):
            i, query = remain_prompts[k]
            res, report = remain_results[k]
            exec_result = res if res else report
            if "pal" in args.prompt_type:
                exec_result = "\\boxed{" + exec_result + "}"
            exec_result = f"\n```output\n{exec_result}\n```\n"
            query += exec_result
            # not end
            if epoch == max_func_call - 1:
                query += "\nReach max function call limit."
            remain_prompts[k] = (i, query)

    # unsolved samples
    print("Unsolved samples:", len(remain_prompts))
    end_prompts.extend(remain_prompts)
    # sort by idx
    end_prompts = sorted(end_prompts, key=lambda x: x[0])

    # remove input_prompt from end_prompt
    codes = []
    assert len(input_prompts) == len(end_prompts)
    for i in range(len(input_prompts)):
        _, end_prompt = end_prompts[i]
        code = end_prompt.split(input_prompts[i])[-1].strip()
        codes.append(code)

    # extract preds
    results = [run_execute(executor, code, args.prompt_type, data_name) for code in codes]
    time_use = time.time() - start_time

    # put results back to examples
    all_samples = []
    for i, sample in enumerate(samples):
        code = codes[i*args.n_sampling: (i+1)*args.n_sampling]
        result = results[i*args.n_sampling: (i+1)*args.n_sampling]
        preds = [item[0] for item in result]
        reports = [item[1] for item in result]

        sample.pop('prompt')
        sample.update({'code': code, 'pred': preds, 'report': reports})
        all_samples.append(sample)

    # add processed samples
    all_samples.extend(processed_samples)
    all_samples, result_json = evaluate(samples=all_samples, data_name=data_name, prompt_type=args.prompt_type, execute=True, split=args.split, model_name_or_path=args.model_name_or_path, data_file=data_file, test_level=args.test_level)
    print(f'dataname:{data_name}    n_sampling:{args.n_sampling}')
    # save outputs
    if len(processed_samples) < len(all_samples) and args.save_outputs:
        save_jsonl(all_samples, out_file)
    
    result_json['time_use_in_second'] = time_use
    result_json['time_use_in_minite'] = f"{int(time_use // 60)}:{int(time_use % 60):02d}"
    input_file = out_file.replace(".jsonl", f"_{args.prompt_type}_metrics.json")
    with open(out_file.replace(".jsonl", f"_{args.prompt_type}_metrics.json"), "w") as f:
        json.dump(result_json, f, indent=4)
    if args.test_level:
        acc_distribution(args, out_file)
        # data_process(args, out_file)
    return result_json

if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    setup(args)

