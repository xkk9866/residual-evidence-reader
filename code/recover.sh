#!/bin/bash
# recovery of the jobs that failed for memory reasons, on a free GPU 6, then the final evaluation
cd /data/yaokang/proofset; PY=.venv/bin/python
export CUDA_VISIBLE_DEVICES=6 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
L=logs/recover.log; echo "[$(date)] start" > $L
H=ckpt/mq_lacr/model.pt
N40=data/proc/openpool_musique_N40.jsonl,data/proc/openpool_hotpotqa_N40.jsonl,data/proc/openpool_2wikimultihopqa_N40.jsonl
( $PY code/train3.py --ckpt $H --out ckpt/mq_lacr_ev --infer $N40 > logs/rec_infh40.log 2>&1; echo "infh40 $?" >> $L ) &
( $PY code/train3.py --ckpt $H --out ckpt/mq_lacr_L320 --doclen 320 --infer data/proc/musique_dev.jsonl,data/proc/openpool_musique_N20.jsonl > logs/rec_L320.log 2>&1; echo "L320 $?" >> $L ) &
( $PY code/train3.py --term text --ckpt ckpt/mq_lact_s43/model_q3.pt --out ckpt/mq_lact_s43_q3 --infer data/proc/musique_dev.jsonl > logs/rec_q3.log 2>&1 && \
  $PY code/train3.py --term text --ckpt ckpt/mq_lact_s43/model_q3.pt --out ckpt/mq_lact_s43_q3 --max_infer 1600 --infer data/proc/musique_full_dev.jsonl >> logs/rec_q3.log 2>&1; echo "q3 $?" >> $L ) &
( bash code/loop_run.sh hotpotqa lach fixed > logs/rec_loop.log 2>&1; bash code/loop_eval.sh hotpotqa >> logs/rec_loop.log 2>&1; echo "loop $?" >> $L ) &
wait
# stage-1 seed 43 models also need MuSiQue-Full for answerability
QA="$PY code/llm_tools.py qamulti"; export GPU_UTIL=0.85
ALL=""; SUB=""
for d in musique hotpotqa 2wikimultihopqa; do
  for t in lac_adaptive lac_fixed lach_adaptive lach_fixed static_adaptive static_fixed chain_adaptive chain_fixed dense_fixed; do ALL="$ALL loop_${d}_${t}_r2|${t}:k4"; done
  for t in static_fixed static_adaptive chain_fixed dense_fixed lach_fixed lach_adaptive; do SUB="$SUB loop_${d}_${t}_r2|${t}:k4"; done
  SUB="$SUB openpool_${d}_N20|lacg:k4 openpool_${d}_N20|jbce3:k4"
done
RTAG=qwen3_8b $QA "musique_dev|lach_dec:dec" "musique_dev|lach_cal_dec:dec" "musique_dev|lact_dec:dec" "musique_dev|lact_cal_dec:dec" \
  "musique_dev|jbce4tuned_dec:dec" "musique_dev|chaintuned_dec:dec" "musique_dev|lact:k2,k4,t512" $ALL > logs/rec_qa_qwen3.log 2>&1; echo "qa_qwen3 $?" >> $L
R2="READER=$HOME/models/Qwen2.5-7B-Instruct-AWQ RTAG=qwen25_7b"
env $R2 $QA "musique_dev|lacg:k4" "musique_dev|jbce3:k4" "musique_dev|jbce4:k4" "musique_dev|chain:k4" "musique_dev|setr:k4" "musique_dev|rankgpt:k4" \
  "musique_dev|beam:k4" "musique_dev|oracle:k4" "musique_dev|fullpool:k4" "musique_dev|lach_cal_dec:dec" "musique_dev|jbce3tuned_dec:dec" $SUB > logs/rec_qa_q25.log 2>&1; echo "qa_q25 $?" >> $L
env $R2 QA_N=2000 $QA "hotpotqa_dev|lacg:k2" "hotpotqa_dev|jbce:k2" "2wiki_dev|lacg:k2" "2wiki_dev|jbce:k2" "hotpotqa_dev|oracle:k2" "2wiki_dev|oracle:k2" >> logs/rec_qa_q25.log 2>&1; echo "qa_q25x $?" >> $L
bash code/final20.sh > logs/final20.log 2>&1; echo "final $?" >> $L
echo "[$(date)] RECOVER_DONE" >> $L
