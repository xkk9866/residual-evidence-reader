#!/bin/bash
# Missing LLM jobs (RankGPT + QA) on GPU 0 or 7, whichever has >= 11 GB free; retries until each job produces output.
cd /data/yaokang/proofset
export HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True GPU_UTIL=0.93
PY=.venv/bin/python
ST=logs/q13_status.log
D=musique_dev
pick() {   # echo a GPU id (0 or 7) with >= 11000 MiB free, waiting if necessary
  while true; do
    for g in 0 7; do
      f=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $g)
      [ "$f" -ge 9000 ] && { echo $g; return; }
    done
    sleep 60
  done
}
job() {   # job <check-file> <log> <cmd...>
  chk=$1; log=$2; shift 2
  while [ ! -f "$chk" ]; do
    g=$(pick); echo "[$(date)] START gpu$g $*" >> $ST
    CUDA_VISIBLE_DEVICES=$g "$@" > $log 2>&1
    echo "[$(date)] END rc=$? $*" >> $ST
    [ -f "$chk" ] || sleep 120
  done
}
job results/scores/${D}_rankgpt_qwen3_8b.pkl logs/rank_mq.log $PY code/llm_tools.py rank $D
$PY code/eval_select.py $D rankgpt=results/scores/${D}_rankgpt_qwen3_8b.pkl:sets >> logs/eval_q12_mq.log 2>&1
job results/qa/${D}_noctx_k2_qwen3_8b.json logs/qa_final_mq.log $PY code/llm_tools.py qa $D lacg:k2,k4,t512 jbce3:k2,k4,t512 chain:k2,k4,t512 beam:k2,k4,t512 q3rr:k2,k4,t512 \
  rankgpt:k2,k4,t512 setr:k2,k4,t512 point:k2,k4,t512 prove2:k2,k4,t512 oracle:k2 fullpool:k2 noctx:k2
export QA_N=2000
for ds in hotpotqa 2wiki; do
  job results/qa/${ds}_dev_oracle_k2_qwen3_8b.json logs/qa_final_${ds}.log $PY code/llm_tools.py qa ${ds}_dev lacg:k2,t384 jbce:k2,t384 beam:k2,t384 q3rr:k2,t384 oracle:k2
done
$PY code/gen_tables.py > logs/gen_tables.log 2>&1
echo "[$(date)] ALL DONE" >> $ST
