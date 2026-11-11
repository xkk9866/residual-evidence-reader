#!/bin/bash
# cross-swap loops (selector vs stopper) and closure_rank inference on GPU5; runs after q31
cd /data/yaokang/proofset; PY=.venv/bin/python
export HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=5 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
st() { echo "$(date) $*" >> logs/q32_status.log; }
while ! grep -q ALL logs/q31_status.log 2>/dev/null; do sleep 120; done
for d in 2wikimultihopqa hotpotqa musique; do for m in static-lacc lacc-anscls; do
  [ -f done/loop_${d}_${m}_adaptive ] || { st loop $d $m; bash code/loop_run.sh $d $m adaptive >> logs/q32_loop.log 2>&1 && touch done/loop_${d}_${m}_adaptive; }
done; bash code/loop_eval.sh $d > logs/loopeval_$d.txt 2>&1; done
while [ ! -f done/tr_laccr_42 ]; do sleep 120; done
[ -f done/inf_laccr ] || { st inf laccr; $PY code/train3.py --objective closure_rank --ckpt ckpt/mq_laccr/model.pt --out ckpt/mq_laccr --infer data/proc/musique_dev.jsonl,data/proc/musique_full_dev.jsonl >> logs/q32_inf.log 2>&1 && touch done/inf_laccr; }
st ALL
