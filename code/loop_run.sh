#!/bin/bash
# loop_run.sh DS METHOD MODE : three retrieval rounds (initial + 2 re-retrievals) for one reader, equal opportunity.
cd /data/yaokang/proofset
DS=$1; M=$2; MM=$2; MODE=$3; PY=.venv/bin/python; TAG=loop_${DS}_${M}_${MODE}
infer() {  # reader inference on pool $1 (SEL-ASK: both readers)
  for M in ${MM//-/ }; do
  case $M in
    lac)    [ -f ckpt/mq_lact/pred_$1.pkl ] || $PY code/train3.py --term text --ckpt ckpt/mq_lact/model.pt --out ckpt/mq_lact --infer data/proc/$1.jsonl ;;
    lach)   [ -f ckpt/mq_lacr_ev/pred_$1.pkl ] || $PY code/train3.py --ckpt ckpt/mq_lacr/model.pt --out ckpt/mq_lacr_ev --infer data/proc/$1.jsonl ;;
    lacc)   [ -f ckpt/mq_lacc/pred_$1.pkl ] || $PY code/train3.py --objective closure --ckpt ckpt/mq_lacc/model.pt --out ckpt/mq_lacc --infer data/proc/$1.jsonl ;;
    delta)  [ -f ckpt/mq_delta/pred_$1.pkl ] || $PY code/train3.py --objective delta --ckpt ckpt/mq_delta/model.pt --out ckpt/mq_delta --infer data/proc/$1.jsonl ;;
    static) [ -f ckpt/mq_jbce3/pred_$1.pkl ] || $PY code/train2.py --head bce --ckpt ckpt/mq_jbce3/model.pt --out ckpt/mq_jbce3 --infer data/proc/$1.jsonl ;;
    chain)  [ -f ckpt/mq_chain/pred_$1.pkl ] || $PY code/chain_ce.py --ckpt ckpt/mq_chain/model.pt --out ckpt/mq_chain --infer data/proc/$1.jsonl ;;
    anscls) [ -f ckpt/mq_anscls/pred_$1.pkl ] || $PY code/train3.py --objective anscls --ckpt ckpt/mq_anscls/model.pt --out ckpt/mq_anscls --infer data/proc/$1.jsonl ;;
    dense)  true ;;
  esac
  done
  M=$MM
  if [ "$MODE" = adaptive ] && { [ $M = static ] || [ $M = chain ]; }; then
    [ -f ckpt/mq_anscls/pred_$1.pkl ] || $PY code/train3.py --objective anscls --ckpt ckpt/mq_anscls/model.pt --out ckpt/mq_anscls --infer data/proc/$1.jsonl
  fi
}
set -e
for R in 0 1 2; do
  P=$([ $R = 0 ] && echo openpool_${DS}_N20 || echo ${TAG}_r$R)
  infer $P
  [ $R -lt 2 ] && $PY code/loop_build.py $DS $M $MODE $R
done
echo LOOP_OK $TAG
