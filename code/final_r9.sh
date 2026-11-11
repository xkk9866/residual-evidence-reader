#!/bin/bash
# after q34/q35: every remaining CPU evaluation of round 9, then the tables
cd /data/yaokang/proofset; PY=.venv/bin/python; D=musique_dev
bash code/eval_r9.sh > logs/eval_r9.log 2>&1
for d in musique hotpotqa 2wikimultihopqa; do $PY code/eval_select.py openpool_${d}_N40 lacc=ckpt/mq_lacc/pred_openpool_${d}_N40.pkl:order; done
for T in 0 1 2 3 4; do $PY code/eval_select.py $D lacc_T$T=ckpt/mq_lacc_T$T/pred_$D.pkl:order; done
$PY code/eval_select.py $D lacc_L320=ckpt/mq_lacc_L320/pred_$D.pkl:order
$PY code/eval_select.py openpool_musique_N20 lacc_L320=ckpt/mq_lacc_L320/pred_openpool_musique_N20.pkl:order
for d in musique hotpotqa 2wikimultihopqa; do
  bash code/loop_eval.sh $d > logs/loopeval_$d.txt 2>&1
  for b in static_fixed chain_fixed dense_fixed; do $PY code/loop_signif.py $d lacc_fixed $b; done
  for b in static_adaptive static_fixed lach_adaptive lacc-anscls_adaptive static-lacc_adaptive; do $PY code/loop_signif.py $d lacc_adaptive $b; done
done > logs/loopsig_r9.log 2>&1
$PY code/gen_tables5.py > logs/gen5.log 2>&1
echo FINAL_R9_DONE >> logs/gen5.log
