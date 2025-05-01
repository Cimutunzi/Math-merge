#!/bin/bash
export ARK_API_KEY="643b29a6-86e2-4854-bbbf-990b5df8ff50"
nohup python  /data/qq/data_process/example/retry_step.py \
    --input "/data/qq/math-evaluation-harness/data/math-merge/train.jsonl" \
    --output "/data/qq/math-evaluation-harness/data/math-merge/step_retry_1.jsonl" \
    --example_1 "$(cat /data/qq/data_process/example/step_1.jsonl)" \
    --example_2 "$(cat /data/qq/data_process/example/step_2.jsonl)" \
    > /data/qq/data_process/example/step_retry.log 2>&1 &

