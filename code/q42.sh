#!/bin/bash
# After local Beam (q40) and IRCoT (q41): swap in the locally trained Beam, re-evaluate, QA, tables
cd /data/yaokang/proofset; PY=.venv/bin/python
export HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
st() { echo "$(date) $*" >> logs/q42_status.log; }
until [ -f done/beam_musique ] && [ -f done/beam_hotpotqa ] && [ -f done/beam_2wiki ] && grep -q ALL logs/q41_status.log 2>/dev/null; do sleep 600; done
for d in musique hotpotqa 2wiki; do
  D=${d}_dev
  [ -f results/scores/${D}_beam_official.pkl ] || cp results/scores/${D}_beam.pkl results/scores/${D}_beam_official.pkl
  cp results/scores/${D}_beamL.pkl results/scores/${D}_beam.pkl
  $PY code/eval_select.py $D beam=results/scores/${D}_beam.pkl:sets beamk=results/scores/${D}_beamK.pkl:sets beamoff=results/scores/${D}_beam_official.pkl:sets
done
st eval
pick() { for g in 5 0; do f=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $g); [ $f -gt 11000 ] && { echo $g; return; }; done; }
CUDA_VISIBLE_DEVICES=5 bash code/eval_r9.sh > logs/eval_r9b.log 2>&1
st eval_r9
until [ -f done/qa42 ]; do
  g=""; while [ -z "$g" ]; do g=$(pick); [ -z "$g" ] && sleep 300; done
  u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g)
  export CUDA_VISIBLE_DEVICES=$g GPU_UTIL=$(python3 -c "print(min(0.95, round(($u + 9500) / 24564, 2)))")
  $PY code/llm_tools.py qamulti "musique_dev|beam:k2,k4,t512,k5" "musique_dev|beamk:k2,k4,t512" "musique_dev|beam_dec:dec" > logs/qa42.log 2>&1 && \
  QA_N=2000 $PY code/llm_tools.py qamulti "hotpotqa_dev|beam:k2,t384,k5" "hotpotqa_dev|beamk:k2,t384" "2wiki_dev|beam:k2,t384,k5" "2wiki_dev|beamk:k2,t384" >> logs/qa42.log 2>&1 && touch done/qa42 || { st retry qa; sleep 300; }
done
st qa
$PY code/gen_tables6.py > logs/gen6.log 2>&1
st ALL
