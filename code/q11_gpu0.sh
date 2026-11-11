#!/bin/bash
# TRAINING queue on GPU 0 (runs in parallel with q12_gpu7.sh, which picks up ckpts via done/tr_<name>).
cd /data/yaokang/proofset
export HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python
ST=logs/q11_status.log
run() { echo "[$(date)] START $*" >> $ST; "$@"; echo "[$(date)] END rc=$? $*" >> $ST; }
mkdir -p done
T3="$PY code/train3.py --data musique --init ckpt/mq_jbce/model.pt --epochs 1"
for v in "abl_noroll --no_roll" "abl_noworld --no_world" "abl_rbce --objective rbce" "mq_lacr_s43 --seed 43" "mq_lacr_s44 --seed 44"; do
  set -- $v; a=$1; shift
  [ -f done/tr_$a ] || { run $T3 --out ckpt/$a "$@" > logs/$a.log 2>&1; touch done/tr_$a; }
done
for s in 43 44; do
  a=mq_jbce3_s$s
  [ -f done/tr_$a ] || { run $PY code/train2.py --data musique --head bce --ckpt ckpt/mq_jbce/model.pt --out ckpt/$a --epochs 1 --seed $s > logs/$a.log 2>&1; touch done/tr_$a; }
done
touch done/TRAIN_ALL
echo "[$(date)] ALL DONE" >> $ST
