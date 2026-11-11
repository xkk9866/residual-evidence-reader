#!/bin/bash
# INFERENCE / EVAL / LLM queue on GPU 7 (another user's process holds ~9.8 GB of GPU 7; vLLM uses GPU_UTIL=0.5).
cd /data/yaokang/proofset
export HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=7 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True GPU_UTIL=0.5
PY=.venv/bin/python
ST=logs/q12_status.log
run() { echo "[$(date)] START $*" >> $ST; "$@"; echo "[$(date)] END rc=$? $*" >> $ST; }
mark() { touch done/$1; }
skip() { [ -f done/$1 ]; }
D=musique_dev; F=musique_full_dev
INF="data/proc/$D.jsonl,data/proc/$F.jsonl"
# 1) abl_prefix (trained under q10)
skip inf_abl_prefix || { run $PY code/train3.py --ckpt ckpt/abl_prefix/model.pt --out ckpt/abl_prefix --infer $INF > logs/abl_prefix_infer.log 2>&1; mark inf_abl_prefix; }
run $PY code/eval_select.py $D abl_prefix=ckpt/abl_prefix/pred_$D.pkl:order abl_mark=ckpt/abl_mark/pred_$D.pkl:order > logs/eval_abl12.log 2>&1
# 2) zero-shot Qwen3-Reranker-4B (int8)
for ds in musique_dev hotpotqa_dev 2wiki_dev; do skip q3rr_$ds || { run $PY code/q3rr.py $ds > logs/q3rr_$ds.log 2>&1; mark q3rr_$ds; }; done
run $PY code/eval_select.py $D q3rr=results/scores/${D}_q3rr.pkl:score > logs/eval_q12_mq.log 2>&1
for ds in hotpotqa 2wiki; do
  run $PY code/eval_select.py ${ds}_dev lacg=ckpt/${ds}_lacr/pred_${ds}_dev.pkl:order jbce=ckpt/${ds}_jbce/pred_${ds}_dev.pkl:score \
    prove2=ckpt/${ds}_v2/pred_${ds}_dev.pkl:circuit beam=results/scores/${ds}_dev_beam.pkl:sets q3rr=results/scores/${ds}_dev_q3rr.pkl:score \
    rerank=results/scores/${ds}_dev_rerank.pkl:score bm25=results/scores/${ds}_dev_bm25.pkl:score > logs/eval_final_${ds}.log 2>&1
done
# 3) LLM: RankGPT + QA
skip rank_mq || { run $PY code/llm_tools.py rank $D > logs/rank_mq.log 2>&1; mark rank_mq; }
run $PY code/eval_select.py $D rankgpt=results/scores/${D}_rankgpt_qwen3_8b.pkl:sets >> logs/eval_q12_mq.log 2>&1
skip qa_mq || { run $PY code/llm_tools.py qa $D lacg:k2,k4,t512 jbce3:k2,k4,t512 chain:k2,k4,t512 beam:k2,k4,t512 q3rr:k2,k4,t512 \
  rankgpt:k2,k4,t512 setr:k2,k4,t512 point:k2,k4,t512 prove2:k2,k4,t512 oracle:k2 fullpool:k2 noctx:k2 > logs/qa_final_mq.log 2>&1; mark qa_mq; }
export QA_N=2000
for ds in hotpotqa 2wiki; do
  skip qa_$ds || { run $PY code/llm_tools.py qa ${ds}_dev lacg:k2,t384 jbce:k2,t384 beam:k2,t384 q3rr:k2,t384 oracle:k2 > logs/qa_final_${ds}.log 2>&1; mark qa_$ds; }
done
unset QA_N
# 4) ablations and seeds trained on GPU 0
for a in abl_noroll abl_noworld abl_rbce mq_lacr_s43 mq_lacr_s44; do
  while [ ! -f done/tr_$a ]; do sleep 60; done
  skip inf_$a || { run $PY code/train3.py --ckpt ckpt/$a/model.pt --out ckpt/$a --infer $INF > logs/${a}_infer.log 2>&1; mark inf_$a; }
  run $PY code/eval_select.py $D $a=ckpt/$a/pred_$D.pkl:order >> logs/eval_abl12.log 2>&1
done
for s in 43 44; do
  a=mq_jbce3_s$s
  while [ ! -f done/tr_$a ]; do sleep 60; done
  skip inf_$a || { run $PY code/train2.py --head bce --ckpt ckpt/$a/model.pt --out ckpt/$a --infer $INF > logs/${a}_infer.log 2>&1; mark inf_$a; }
  run $PY code/eval_select.py $D jbce3_s$s=ckpt/$a/pred_$D.pkl:score >> logs/eval_abl12.log 2>&1
done
# 5) answerability, significance, fairness, tables
run $PY code/ans_eval.py lacg=ckpt/mq_lacr/pred_$F.pkl:order anscls=ckpt/mq_anscls/pred_$F.pkl:cls prove2=ckpt/mq_v2/pred_$F.pkl:circuit \
  jbce3=ckpt/mq_jbce3/pred_$F.pkl:score abl_noworld=ckpt/abl_noworld/pred_$F.pkl:order abl_mark=ckpt/abl_mark/pred_$F.pkl:order \
  abl_prefix=ckpt/abl_prefix/pred_$F.pkl:order abl_noroll=ckpt/abl_noroll/pred_$F.pkl:order \
  mq_lacr_s43=ckpt/mq_lacr_s43/pred_$F.pkl:order mq_lacr_s44=ckpt/mq_lacr_s44/pred_$F.pkl:order > logs/ans_final.log 2>&1
run $PY code/ans_sig.py > logs/ans_sig.log 2>&1
run $PY code/stop_fair.py > logs/stop_fair.log 2>&1
run $PY code/signif.py $D lacg jbce3 > logs/signif_final.log 2>&1
for b in beam chain q3rr abl_rbce abl_mark; do run $PY code/signif.py $D lacg $b >> logs/signif_final.log 2>&1; done
for ds in hotpotqa 2wiki; do for b in jbce beam; do run $PY code/signif.py ${ds}_dev lacg $b k2,k3,k4,t256,t384,t512 >> logs/signif_final.log 2>&1; done; done
run $PY code/gen_tables.py > logs/gen_tables.log 2>&1
touch done/ALL
echo "[$(date)] ALL DONE" >> $ST
