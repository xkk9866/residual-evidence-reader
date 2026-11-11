#!/bin/bash
# GPU6 only. Point-process residual reader (LACUNA-PP): train + infer + eval. resumable via done/
cd /data/yaokang/proofset
export HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=6 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python
ST=logs/q14_status.log
run() { echo "[$(date)] START $*" >> $ST; "$@"; echo "[$(date)] END rc=$? $*" >> $ST; }
mkdir -p done
a=mq_lacp
[ -f done/tr_$a ] || { run $PY code/train3.py --data musique --init ckpt/mq_jbce/model.pt --epochs 1 --objective point --out ckpt/$a > logs/$a.log 2>&1; touch done/tr_$a; }
[ -f done/inf_$a ] || { run $PY code/train3.py --objective point --ckpt ckpt/$a/model.pt --out ckpt/$a --infer data/proc/musique_dev.jsonl,data/proc/musique_full_dev.jsonl > logs/inf_$a.log 2>&1; touch done/inf_$a; }
run $PY code/eval_select.py musique_dev lacp=ckpt/$a/pred_musique_dev.pkl:order
run $PY code/ans_eval.py lacp=ckpt/$a/pred_musique_full_dev.pkl:order
run $PY code/diag_stop.py ckpt/$a/pred_musique_dev.pkl
echo "[$(date)] ALL DONE" >> $ST
