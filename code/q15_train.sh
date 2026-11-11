#!/bin/bash
# GPU6 TRAINING lane (serial). q16_infer.sh picks up checkpoints via done/tr_<name>.
cd /data/yaokang/proofset
export HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=6 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python
ST=logs/q15_status.log
run() { echo "[$(date)] START $*" >> $ST; "$@"; echo "[$(date)] END rc=$? $*" >> $ST; }
mkdir -p done
while [ ! -f done/tr_mq_lacp ]; do sleep 30; done
# static reader trained for 2 more epochs (>= LACUNA's stage-2 wall clock): time-matched control
a=mq_jbce4
[ -f done/tr_$a ] || { run $PY code/train2.py --data musique --head bce --ckpt ckpt/mq_jbce/model.pt --out ckpt/$a --epochs 2 > logs/$a.log 2>&1; touch done/tr_$a; }
for ds in hotpotqa 2wiki; do
  a=${ds}_lacp
  [ -f done/tr_$a ] || { run $PY code/train3.py --data $ds --objective point --init ckpt/${ds}_jbce/model.pt --out ckpt/$a --epochs 1 --max_train 20000 > logs/$a.log 2>&1; touch done/tr_$a; }
done
T3="$PY code/train3.py --data musique --objective point --init ckpt/mq_jbce/model.pt --epochs 1"
for v in "mq_lacp_s43 --seed 43" "mq_lacp_s44 --seed 44" "pabl_ce1 --loc ce1" "pabl_mark --layout mark" "pabl_noworld --no_world" "pabl_noroll --no_roll" "pabl_prefix --states prefix"; do
  set -- $v; a=$1; shift
  [ -f done/tr_$a ] || { run $T3 --out ckpt/$a "$@" > logs/$a.log 2>&1; touch done/tr_$a; }
done
touch done/TRAIN15_ALL
echo "[$(date)] ALL DONE" >> $ST
