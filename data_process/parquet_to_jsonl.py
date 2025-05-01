import pandas as pd
import pyarrow.parquet as pq

# 读取Parquet文件
file = '/data/qq/data/hg-MATH/algebra/train-00000-of-00001.parquet'
table = pq.read_table(file)
df = table.to_pandas()

# 将PyArrow Table转换为JSONL字符串
jsonl_strings = df.to_json(orient='records', lines=True)

# 写入JSONL文件
with open('/data/qq/data/hg-MATH/algebra/train.jsonl', 'w') as f:
    f.write(jsonl_strings)