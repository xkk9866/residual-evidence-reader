#!/bin/bash
# remaining recovery steps; every step waits until GPU 6 or 7 has enough free memory (other users share them)
cd /data/yaokang/proofset; PY=.venv/bin/python
export HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
L=logs/recover.log
pickgpu() {  # $1 = MiB needed; sets CUDA_VISIBLE_DEVICES and FREE
  while true; do
    for g in 6 7; do
      FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $g)
      if [ "$FREE" -ge "$1" ]; then export CUDA_VISIBLE_DEVICES=$g; return; fi
    done
    sleep 60
  done
}
while pgrep -f "out ckpt/mq_lacr_ev --infer data/proc/openpool_musique_N40" > /dev/null; do sleep 30; done
H=ckpt/mq_lacr/model.pt
pickgpu 9000; $PY code/train3.py --ckpt $H --out ckpt/mq_lacr_L320 --doclen 320 --infer data/proc/musique_dev.jsonl,data/proc/openpool_musique_N20.jsonl > logs/rec_L320.log 2>&1; echo "L320 $?" >> $L
pickgpu 6000; ( $PY code/train3.py --term text --ckpt ckpt/mq_lact_s43/model_q3.pt --out ckpt/mq_lact_s43_q3 --infer data/proc/musique_dev.jsonl && \
  $PY code/train3.py --term text --ckpt ckpt/mq_lact_s43/model_q3.pt --out ckpt/mq_lact_s43_q3 --max_infer 1600 --infer data/proc/musique_full_dev.jsonl ) > logs/rec_q3.log 2>&1; echo "q3 $?" >> $L
QA="$PY code/llm_tools.py qamulti"
ALL=""; SUB=""
for d in musique hotpotqa 2wikimultihopqa; do
  for t in lac_adaptive lac_fixed lach_adaptive lach_fixed static_adaptive static_fixed chain_adaptive chain_fixed dense_fixed; do ALL="$ALL loop_${d}_${t}_r2|${t}:k4"; done
  for t in static_fixed static_adaptive chain_fixed dense_fixed lach_fixed lach_adaptive; do SUB="$SUB loop_${d}_${t}_r2|${t}:k4"; done
  SUB="$SUB openpool_${d}_N20|lacg:k4 openpool_${d}_N20|jbce3:k4"
done
runqa() {  # tag reader args...   (qamulti skips results that already exist, so a failed attempt is simply repeated)
  tag=$1; rd=$2; shift 2
  for a in 1 2 3 4 5 6; do
    pickgpu 12000; export GPU_UTIL=$(python3 -c "print(round(min(0.85,($FREE-1500)/24564),2))")
    env READER=$rd RTAG=$tag $QA "$@" >> logs/rec_qa_$tag.log 2>&1 && { echo "qa_$tag ok" >> $L; return; }
    sleep 120
  done
  echo "qa_$tag FAILED" >> $L
}
runqa qwen3_8b $HOME/models/Qwen3-8B-AWQ "musique_dev|lach_dec:dec" "musique_dev|lach_cal_dec:dec" "musique_dev|lact_dec:dec" "musique_dev|lact_cal_dec:dec" \
  "musique_dev|jbce4tuned_dec:dec" "musique_dev|chaintuned_dec:dec" "musique_dev|lact:k2,k4,t512" $ALL
runqa qwen25_7b $HOME/models/Qwen2.5-7B-Instruct-AWQ "musique_dev|lacg:k4" "musique_dev|jbce3:k4" "musique_dev|jbce4:k4" "musique_dev|chain:k4" "musique_dev|setr:k4" \
  "musique_dev|rankgpt:k4" "musique_dev|beam:k4" "musique_dev|oracle:k4" "musique_dev|fullpool:k4" "musique_dev|lach_cal_dec:dec" "musique_dev|jbce3tuned_dec:dec" $SUB
export QA_N=2000
runqa qwen25_7b $HOME/models/Qwen2.5-7B-Instruct-AWQ "hotpotqa_dev|lacg:k2" "hotpotqa_dev|jbce:k2" "2wiki_dev|lacg:k2" "2wiki_dev|jbce:k2" "hotpotqa_dev|oracle:k2" "2wiki_dev|oracle:k2"
unset QA_N
bash code/final20.sh > logs/final20.log 2>&1; echo "final $?" >> $L
echo "[$(date)] RECOVER_DONE" >> $L
