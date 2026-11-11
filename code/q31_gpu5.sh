#!/bin/bash
# round 9 inference/loops on GPU5 (waits for GPU0 checkpoints)
cd /data/yaokang/proofset; PY=.venv/bin/python; mkdir -p done
export HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=5 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
MQ=data/proc/musique_dev.jsonl,data/proc/musique_full_dev.jsonl
OP=data/proc/openpool_musique_N20.jsonl,data/proc/openpool_hotpotqa_N20.jsonl,data/proc/openpool_2wikimultihopqa_N20.jsonl
st() { echo "$(date) $*" >> logs/q31_status.log; }
for s in 42 43 44; do
  O=ckpt/mq_lacc$([ $s = 42 ] || echo _s$s)
  while [ ! -f done/tr_lacc_$s ]; do sleep 60; done
  [ -f done/inf_lacc_$s ] || { st inf $O; $PY code/train3.py --objective closure --ckpt $O/model.pt --out $O --infer $MQ >> logs/q31_inf.log 2>&1 && touch done/inf_lacc_$s; }
  if [ $s = 42 ]; then
    [ -f done/inf0_lacc ] || { st inf0; $PY code/train3.py --objective closure --ckpt $O/model.pt --out $O --infer $OP >> logs/q31_inf.log 2>&1 && touch done/inf0_lacc; }
    for d in musique hotpotqa 2wikimultihopqa; do for mode in adaptive fixed; do
      [ -f done/loop_${d}_lacc_$mode ] || { st loop $d $mode; bash code/loop_run.sh $d lacc $mode >> logs/q31_loop.log 2>&1 && touch done/loop_${d}_lacc_$mode; }
    done; bash code/loop_eval.sh $d > logs/loopeval_$d.txt 2>&1; done
  fi
done
st ALL
