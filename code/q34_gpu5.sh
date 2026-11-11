#!/bin/bash
# lacc: 40-candidate pools, read-count curve, 320-token passages (GPU5)
cd /data/yaokang/proofset; PY=.venv/bin/python
export HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=5 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
st() { echo "$(date) $*" >> logs/q34_status.log; }
C="$PY code/train3.py --objective closure --ckpt ckpt/mq_lacc/model.pt"
N40=data/proc/openpool_musique_N40.jsonl,data/proc/openpool_hotpotqa_N40.jsonl,data/proc/openpool_2wikimultihopqa_N40.jsonl
[ -f done/lacc_n40 ] || { st n40; $C --out ckpt/mq_lacc --infer $N40 >> logs/q34.log 2>&1 && touch done/lacc_n40; }
for T in 0 1 2 3 4; do
  [ -f done/lacc_T$T ] || { st T$T; mkdir -p ckpt/mq_lacc_T$T; $C --out ckpt/mq_lacc_T$T --steps $T --infer data/proc/musique_dev.jsonl > logs/mq_lacc_T$T.log 2>&1 && touch done/lacc_T$T; }
done
[ -f done/lacc_L320 ] || { st L320; $C --out ckpt/mq_lacc_L320 --doclen 320 --infer data/proc/musique_dev.jsonl,data/proc/openpool_musique_N20.jsonl >> logs/q34.log 2>&1 && touch done/lacc_L320; }
st ALL
