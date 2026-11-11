cd /data/yaokang/proofset; PY=.venv/bin/python; L=logs/recover.log
export HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=6 GPU_UTIL=0.8
$PY code/eval_select.py openpool_musique_N20 lach=ckpt/mq_lacr_ev/pred_openpool_musique_N20.pkl:order > /dev/null 2>&1
SUB=""; for d in musique hotpotqa 2wikimultihopqa; do for t in static_fixed static_adaptive chain_fixed dense_fixed lach_fixed lach_adaptive; do SUB="$SUB loop_${d}_${t}_r2|${t}:k4"; done; SUB="$SUB openpool_${d}_N20|lacg:k4 openpool_${d}_N20|jbce3:k4"; done
R="READER=$HOME/models/Qwen3-4B-AWQ RTAG=qwen3_4b"
env $R $PY code/llm_tools.py qamulti "musique_dev|lacg:k4" "musique_dev|jbce3:k4" "musique_dev|jbce4:k4" "musique_dev|chain:k4" "musique_dev|setr:k4" "musique_dev|rankgpt:k4" "musique_dev|beam:k4" "musique_dev|oracle:k4" "musique_dev|fullpool:k4" "musique_dev|lach_cal_dec:dec" "musique_dev|jbce3tuned_dec:dec" $SUB > logs/rec_qa_q34.log 2>&1; echo "qa_q34 $?" >> $L
env $R QA_N=2000 $PY code/llm_tools.py qamulti "hotpotqa_dev|lacg:k2" "hotpotqa_dev|jbce:k2" "2wiki_dev|lacg:k2" "2wiki_dev|jbce:k2" "hotpotqa_dev|oracle:k2" "2wiki_dev|oracle:k2" >> logs/rec_qa_q34.log 2>&1; echo "qa_q34x $?" >> $L
$PY code/gen_tables4.py 2> logs/gen_tables.err > /dev/null; echo "[$(date)] RECOVER4_DONE" >> $L
