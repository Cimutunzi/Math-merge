import json

input_file = '/data/qq/math-evaluation-harness/data/math/raw_hard_v1.jsonl'
output_file = '/data/qq/math-evaluation-harness/data/math/hard_v1.jsonl'

with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
    for idx, line in enumerate(infile):
        data = json.loads(line)
        data['idx'] = idx
        outfile.write(json.dumps(data) + '\n')