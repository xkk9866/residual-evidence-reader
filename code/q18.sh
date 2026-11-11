#!/bin/bash
# after q17 GPU part: evaluations (CPU) + all LLM QA (Qwen3-8B-AWQ int4, vLLM) on GPU6. resumable.
cd /data/yaokang/proofset
export HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=6 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True GPU_UTIL=0.85
PY=.venv/bin/python
ST=logs/q18_status.log
run() { echo "[$(date)] START $*" >> $ST; "$@"; echo "[$(date)] END rc=$? $*" >> $ST; }
mkdir -p done
while [ ! -f done/Q17_GPU ]; do sleep 60; done
D=musique_dev; F=musique_full_dev
run $PY code/eval_select.py $D jbce4=ckpt/mq_jbce4/pred_$D.pkl:score abl_ce1=ckpt/abl_ce1/pred_$D.pkl:order lacp=ckpt/mq_lacp/pred_$D.pkl:order
run $PY code/ans_eval.py abl_ce1=ckpt/abl_ce1/pred_$F.pkl:order lacp=ckpt/mq_lacp/pred_$F.pkl:order jbce4=ckpt/mq_jbce4/pred_$F.pkl:score
run $PY code/halt_stats.py
run $PY code/beam_matched.py lacg=ckpt/mq_lacr/pred_$D.pkl jbce3=ckpt/mq_jbce3/pred_$D.pkl jbce4=ckpt/mq_jbce4/pred_$D.pkl chain=ckpt/mq_chain/pred_$D.pkl
for b in jbce4 abl_ce1; do run $PY code/signif.py $D lacg $b >> logs/signif_q18.log 2>&1; done
# open corpus: dense order = pool order
$PY - <<PYEOF
import json, pickle, numpy as np
for ds in ['musique', 'hotpotqa', '2wikimultihopqa']:
    R = [json.loads(l) for l in open(f'data/proc/openpool_{ds}_N20.jsonl')]
    pickle.dump([{'rho': -np.arange(len(r['docs']), dtype=float)} for r in R], open(f'results/scores/openpool_{ds}_N20_dense.pkl', 'wb'))
PYEOF
for ds in musique hotpotqa 2wikimultihopqa; do
  P=openpool_${ds}_N20
  run $PY code/eval_select.py $P dense=results/scores/${P}_dense.pkl:score lacg=ckpt/mq_lacr/pred_$P.pkl:order jbce3=ckpt/mq_jbce3/pred_$P.pkl:score chain=ckpt/mq_chain/pred_$P.pkl:order
  for m in adaptive fixed; do run $PY code/eval_select.py openloop_${ds}_${m}2 lacg=ckpt/mq_lacr/pred_openloop_${ds}_${m}2.pkl:order; done
done
# ---- LLM QA ----
[ -f done/qa18_mq ] || { run $PY code/llm_tools.py qa $D lacg_dec:dec jbce3tuned_dec:dec lacp_dec:dec beam_dec:dec jbce4:k2,k4,t512 > logs/qa18_mq.log 2>&1; touch done/qa18_mq; }
for ds in musique hotpotqa 2wikimultihopqa; do
  P=openpool_${ds}_N20
  [ -f done/qa18_$ds ] || { run $PY code/llm_tools.py qa $P dense:k4 lacg:k4 jbce3:k4 chain:k4 > logs/qa18_$P.log 2>&1
    for m in adaptive fixed; do run $PY code/llm_tools.py qa openloop_${ds}_${m}2 lacg:k4 >> logs/qa18_$P.log 2>&1; done; touch done/qa18_$ds; }
done
touch done/Q18_ALL
echo "[$(date)] ALL DONE" >> $ST
