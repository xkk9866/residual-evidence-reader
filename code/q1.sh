#!/bin/bash
# GPU-6 job queue (sequential): MuSiQue PROVE + pointwise, then dev / MuSiQue-Full inference
cd /data/yaokang/proofset
export HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=6 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python
$PY code/train.py --data musique --head circuit --out ckpt/musique_prove --epochs 2 --qpb 2 --accum 4 > logs/musique_prove.log 2>&1
$PY code/train.py --data musique --head circuit --ckpt ckpt/musique_prove/model.pt --out ckpt/musique_prove --infer data/proc/musique_dev.jsonl,data/proc/musique_full_dev.jsonl > logs/musique_prove_infer.log 2>&1
$PY code/train.py --data musique --head pointwise --out ckpt/musique_point --epochs 2 --qpb 2 --accum 4 > logs/musique_point.log 2>&1
$PY code/train.py --data musique --head pointwise --ckpt ckpt/musique_point/model.pt --out ckpt/musique_point --infer data/proc/musique_dev.jsonl,data/proc/musique_full_dev.jsonl > logs/musique_point_infer.log 2>&1
for d in musique_dev musique_full_dev hotpotqa_dev 2wiki_dev; do $PY code/baselines_zs.py $d dense,rerank > logs/zs_$d.log 2>&1; done
