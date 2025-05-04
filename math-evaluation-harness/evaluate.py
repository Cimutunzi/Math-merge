import argparse
import numpy as np
from tqdm import tqdm
from pebble import ProcessPool
from concurrent.futures import TimeoutError
import json
from grader import *

from parser import *
from utils import load_jsonl
from python_executor import PythonExecutor


def evaluate(data_name, prompt_type, split, model_name_or_path, data_file, test_level, samples: list=None, file_path: str=None, max_num_samples=None, execute=False):
    print('许愿每天发财！')
    assert samples or file_path, "samples or file_path must be provided"
    if not samples:
        samples = list(load_jsonl(file_path))
    # dedup by idx
    if 'idx' in samples[0]:
        samples = {sample['idx']: sample for sample in samples}.values()
        samples = sorted(samples, key=lambda x: x['idx']) 
    else:
        samples = [dict(idx=idx, **sample) for idx, sample in enumerate(samples)]

    if max_num_samples:
        print(f"max_num_samples: {max_num_samples} / {len(samples)}")
        samples = samples[:max_num_samples]
    
    # parse gt
    for sample in samples:
        sample['gt_cot'], sample['gt'] = parse_ground_truth(sample, data_name)

    # execute
    if ('pred' not in samples[0]) or execute:
        if "pal" in prompt_type:
            executor = PythonExecutor(get_answer_expr="solution()")
        else:
            executor = PythonExecutor(get_answer_from_stdout=True)

        for sample in tqdm(samples, desc="Execute"):
            sample['pred'] = []
            sample['report'] = []
            for code in sample['code']:
                pred, report = run_execute(executor, code, prompt_type, data_name, execute=True)
                sample['pred'].append(pred)
                sample['report'].append(report)

    params = [(idx, pred, sample['gt']) for idx, sample in enumerate(samples) for pred in sample['pred']]

    scores = []
    timeout_cnt = 0 

    with ProcessPool() as pool:
        future = pool.map(math_equal_process, params, timeout=3)
        iterator = future.result()
        with tqdm(total=len(samples), desc="Evaluate") as progress_bar:
            while True:
                try:
                    result = next(iterator)
                    scores.append(result)
                except StopIteration:
                    break
                except TimeoutError as error:
                    print(error)
                    scores.append(False)
                    timeout_cnt += 1
                except Exception as error:
                    print(error.traceback)
                    exit()
                progress_bar.update(1) 

    idx = 0
    score_mat = []
    pass_mat = []
    major_mat = []
    for sample in samples:
        sample['score'] = scores[idx: idx+len(sample['pred'])]
        assert len(sample['score']) == len(sample['pred'])
        score_mat.append(sample['score'])
        idx += len(sample['pred'])

    max_len = max([len(s) for s in score_mat])

    for i, s in enumerate(score_mat):
        if len(s) < max_len:
            score_mat[i] = s + [s[-1]] * (max_len - len(s)) # pad

    pass_mat = np.array(score_mat)
    pass_mat[pass_mat.any(axis=1)] = 1
    print(pass_mat)

    major_mat = np.array(score_mat)
    major_mat = (np.sum(major_mat == 1, axis=1) >= major_mat.shape[1] / 2).astype(int)[:, np.newaxis] * np.ones(major_mat.shape[1])
    print(major_mat)

    # output mean of each column of scores
    col_means= np.array(score_mat).mean(axis=0)
    mean_score = list(np.round(col_means * 100, decimals=1))

    pass_col_means = np.array(pass_mat).mean(axis=0)
    pass_means = list(np.round(pass_col_means * 100, decimals=1))

    major_col_means = np.array(major_mat).mean(axis=0)
    major_means = list(np.round(major_col_means * 100, decimals=1))

    result_json = {
        "num_samples": len(samples),
        "num_scores": len(scores),
        "timeout_samples": timeout_cnt,
        "empty_samples": len([s for s in samples if not s['pred'][-1]]),
        "acc": mean_score[0],
        "pass@k": pass_means[0],
        "major@k": major_means[0]
    }


    if test_level:
        passed_samples = []
        for sample in samples:
            score = sample['score']
            # 设置 level
            if any(score[:1]):  
                sample['test_level'] = 0
            elif any(score[:4]):
                sample['test_level'] = 1
            elif any(score[:16]):
                sample['test_level'] = 2
            elif any(score[:64]):
                sample['test_level'] = 3
            else:
                sample['test_level'] = 4
            
            # 将通过的样本添加到 passed_samples 中
            passed_samples.append(sample)

        data1 = passed_samples  # 包含 idx 和 level
        if data_file.endswith('.jsonl'):
            data2 = load_jsonl(data_file)  # 从 jsonl 文件加载
        else:
            data2 = load_json(data_file)  # 普通 json 文件加载  # 包含 idx 和其他字段

        # 将 data2 根据 idx 建立一个字典，方便查找
        data2_dict = {item['idx']: item for item in data2}

        # 合并数据
        merged_data = []
        for item1 in data1:
            idx = item1['idx']
            if idx in data2_dict:
                merged_item = {**item1, **data2_dict[idx]}  # 合并
                merged_data.append(merged_item)
            # 将通过的样本保存到文件
        out_file = f'{model_name_or_path}/{data_name}_{split}_{prompt_type}_level.jsonl'
        with open(out_file, 'w') as f:
            for item in merged_data:
                f.write(json.dumps(item) + '\n')

        print(f"Successfully saved {len(passed_samples)} passed samples with 'level' to '{out_file}'.")



    # each type score
    if "type" in samples[0]:
        type_scores = {}
        for sample in samples:
            if sample['type'] not in type_scores:
                type_scores[sample['type']] = []
            type_scores[sample['type']].append(sample['score'][-1])
        type_scores = {k: np.round(np.array(v).mean() * 100, decimals=1) for k, v in type_scores.items()}
        type_scores = {k: v for k, v in sorted(type_scores.items(), key=lambda item: item[0])}
        result_json['type_acc'] = type_scores

    print(result_json)
    return samples, result_json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", type=str, default="math")
    parser.add_argument("--prompt_type", type=str, default="tool-integrated")
    parser.add_argument("--file_path", type=str, default=None, required=True)
    parser.add_argument("--max_num_samples", type=int, default=None)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    return args

def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)
def load_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line.strip()) for line in f]
    
if __name__ == "__main__":
    print('许愿每天发财！')
    args = parse_args()
    evaluate(data_name=args.data_name, prompt_type=args.prompt_type, file_path=args.file_path,
             max_num_samples=args.max_num_samples, execute=args.execute)
