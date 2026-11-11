#!/bin/bash
# GPU-6 queue part 2 (waits for q1): HotpotQA / 2Wiki PROVE + pointwise and their inference
cd /data/yaokang/proofset
while pgrep -f "code/q1.sh" > /dev/null; do sleep 60; done
export HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=6 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python
for ds in hotpotqa 2wiki; do
  $PY code/train.py --data $ds --head circuit --out ckpt/${ds}_prove --epochs 1 --max_train 30000 --qpb 3 --accum 3 > logs/${ds}_prove.log 2>&1
  $PY code/train.py --data $ds --head circuit --ckpt ckpt/${ds}_prove/model.pt --out ckpt/${ds}_prove --infer data/proc/${ds}_dev.jsonl > logs/${ds}_prove_infer.log 2>&1
  $PY code/train.py --data $ds --head pointwise --out ckpt/${ds}_point --epochs 1 --max_train 30000 --qpb 3 --accum 3 > logs/${ds}_point.log 2>&1
  $PY code/train.py --data $ds --head pointwise --ckpt ckpt/${ds}_point/model.pt --out ckpt/${ds}_point --infer data/proc/${ds}_dev.jsonl > logs/${ds}_point_infer.log 2>&1
done
