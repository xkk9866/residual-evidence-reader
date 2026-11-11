#!/bin/bash
# factorial ablation: on-policy states with the decoupled loss only (closure_rank); after q30
cd /data/yaokang/proofset; PY=.venv/bin/python
export HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
while ! grep -q ALL logs/q30_status.log 2>/dev/null; do sleep 120; done
[ -f done/tr_laccr_42 ] || { $PY code/train3.py --data musique --init ckpt/mq_jbce/model.pt --epochs 1 --objective closure_rank --out ckpt/mq_laccr > logs/mq_laccr.log 2>&1 && touch done/tr_laccr_42; }
echo ALL >> logs/q33_status.log
