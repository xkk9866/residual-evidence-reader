#!/bin/bash
# round q21: event-calibrated residual likelihood (3 seeds), clean timing, then loops/QA for the new model
cd /data/yaokang/proofset; PY=.venv/bin/python; L=logs/q21_status.log
export HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
MQ=data/proc/musique_dev.jsonl,data/proc/musique_full_dev.jsonl
gpu() { while true; do for g in 6 7; do f=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $g); [ $f -ge $1 ] && { echo $g; return; }; done; sleep 60; done; }
tr() { s=$1; g=$(gpu 13500); CUDA_VISIBLE_DEVICES=$g $PY code/train3.py --data musique --init ckpt/mq_jbce/model.pt --epochs 1 --objective residual_ev --seed $s --out ckpt/mq_lace_s$s > logs/mq_lace_s$s.log 2>&1 && \
  CUDA_VISIBLE_DEVICES=$g $PY code/train3.py --objective residual_ev --ckpt ckpt/mq_lace_s$s/model.pt --out ckpt/mq_lace_s$s --infer $MQ >> logs/mq_lace_s$s.log 2>&1; echo "lace_s$s $?" >> $L; }
tr 42 & sleep 300; tr 43 & wait; tr 44
echo "[$(date)] Q21_TRAIN_DONE" >> $L
