#!/bin/bash
# round 9: on-policy closure learning (train3 --objective closure), 3 seeds, trained on GPU0
cd /data/yaokang/proofset; PY=.venv/bin/python; mkdir -p done
export HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for s in 42 43 44; do
  O=ckpt/mq_lacc$([ $s = 42 ] || echo _s$s)
  [ -f done/tr_lacc_$s ] && continue
  echo "$(date) start $O" >> logs/q30_status.log
  $PY code/train3.py --data musique --init ckpt/mq_jbce/model.pt --epochs 1 --objective closure --seed $s --out $O > logs/$(basename $O).log 2>&1 && touch done/tr_lacc_$s
  echo "$(date) end $O rc=$?" >> logs/q30_status.log
done
echo ALL >> logs/q30_status.log
