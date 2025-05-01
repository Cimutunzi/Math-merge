import json
import random
from datasets import load_dataset
from tqdm import tqdm

# Load dataset
dataset = load_dataset('nvidia/OpenMathInstruct-2', split='train_1M')

# Sample 20,000 random samples from the dataset
sampled_data = random.sample(dataset, 20000)

print("Converting sampled dataset to jsonl format")
output_file = "/data/qq/data/openmathinstruct2.jsonl"
with open(output_file, 'w', encoding='utf-8') as f:
    for item in tqdm(sampled_data):
        f.write(json.dumps(item, ensure_ascii=False) + '\n')

print(f"Conversion complete. Output saved as {output_file}")
