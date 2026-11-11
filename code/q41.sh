#!/bin/bash
# IRCoT baseline (Qwen3-8B-AWQ + bge-large) on the three open corpora; takes whichever of GPU 0/5 has room first
cd /data/yaokang/proofset; PY=.venv/bin/python
export HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
st() { echo "$(date) $*" >> logs/q41_status.log; }
pick() { for g in 5 0; do f=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $g); [ $f -gt 11000 ] && { echo $g; return; }; done; }
for d in musique hotpotqa 2wikimultihopqa; do
  until [ -f done/ircot_$d ]; do
    g=""; while [ -z "$g" ]; do g=$(pick); [ -z "$g" ] && sleep 300; done
    u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g)
    export CUDA_VISIBLE_DEVICES=$g GPU_UTIL=$(python3 -c "print(min(0.95, round(($u + 9500) / 24564, 2)))")
    $PY code/ircot.py $d > logs/ircot_$d.log 2>&1 && touch done/ircot_$d || { st retry $d; sleep 300; }
  done; st ircot $d
done
st ALL
