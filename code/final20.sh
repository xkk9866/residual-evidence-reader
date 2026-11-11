#!/bin/bash
# all CPU evaluation of round q20 (after every GPU job): selection metrics, answerability, self-termination, significance
cd /data/yaokang/proofset; PY=.venv/bin/python; E=code/eval_select.py; D=musique_dev; F=musique_full_dev
$PY $E $D lach=ckpt/mq_lacr_ev/pred_$D.pkl:order lact=ckpt/mq_lact/pred_$D.pkl:order lact_s43=ckpt/mq_lact_s43/pred_$D.pkl:order \
  lacr_f43=ckpt/mq_lacr_f43/pred_$D.pkl:order jbce3_f43=ckpt/mq_jbce3_f43/pred_$D.pkl:score \
  lach_L320=ckpt/mq_lacr_L320/pred_$D.pkl:order lact_L320=ckpt/mq_lact_L320/pred_$D.pkl:order jbce3_L320=ckpt/mq_jbce3_L320/pred_$D.pkl:score \
  $(for T in 0 1 2 3 4; do echo lach_T$T=ckpt/mq_lacr_T$T/pred_$D.pkl:order lact_T$T=ckpt/mq_lact_T$T/pred_$D.pkl:order; done) \
  $(for q in 1 2 3 4; do echo lact_s43_q$q=ckpt/mq_lact_s43_q$q/pred_$D.pkl:order; done)
$PY code/ans_eval.py lach=ckpt/mq_lacr_ev/pred_$F.pkl:order lact=ckpt/mq_lact/pred_$F.pkl:order lact_s43=ckpt/mq_lact_s43/pred_$F.pkl:order \
  lacr_f43=ckpt/mq_lacr_f43/pred_$F.pkl:order jbce3_f43=ckpt/mq_jbce3_f43/pred_$F.pkl:score
$PY code/halt_stats2.py > /dev/null
FE=""
for n in lach lach_cal lact lact_cal abl_noroll abl_ce1 abl_noworld; do
  case $n in lach*) p=ckpt/mq_lacr_ev;; lact*) p=ckpt/mq_lact;; *) p=ckpt/$n;; esac
  FE="$FE $n=$p/pred_$D.pkl@results/sel/${D}_${n}_dec.json"
done
$PY code/first_error.py $FE > logs/first_error.log 2>&1
$PY code/beam_matched.py lach=ckpt/mq_lacr/pred_$D.pkl lact=ckpt/mq_lact/pred_$D.pkl jbce3=ckpt/mq_jbce3/pred_$D.pkl jbce4=ckpt/mq_jbce4/pred_$D.pkl chain=ckpt/mq_chain/pred_$D.pkl > /dev/null
for b in jbce3 jbce4 chain lact abl_ce1 abl_noroll abl_noworld abl_mark abl_prefix abl_rbce beam; do $PY code/signif.py $D lach $b k2,k3,k4,k5,t384,t512,t768,t1024; done > logs/signif_q20.log 2>&1
$PY code/signif.py $D lacr_f43 jbce3_f43 k4,k5,t768 >> logs/signif_q20.log 2>&1
# open pools: 40 candidates and 320-token passages
for d in musique hotpotqa 2wikimultihopqa; do P=openpool_${d}_N40
  $PY -c "import json,pickle,numpy as np; R=[json.loads(l) for l in open('data/proc/$P.jsonl')]; pickle.dump([{'rho':-np.arange(len(r['docs']),dtype=float)} for r in R],open('results/scores/${P}_dense.pkl','wb'))"
  $PY $E $P dense=results/scores/${P}_dense.pkl:score lach=ckpt/mq_lacr_ev/pred_$P.pkl:order lact=ckpt/mq_lact/pred_$P.pkl:order jbce3=ckpt/mq_jbce3/pred_$P.pkl:score chain=ckpt/mq_chain/pred_$P.pkl:order
done
$PY $E openpool_musique_N20 lach_L320=ckpt/mq_lacr_L320/pred_openpool_musique_N20.pkl:order lact_L320=ckpt/mq_lact_L320/pred_openpool_musique_N20.pkl:order jbce3_L320=ckpt/mq_jbce3_L320/pred_openpool_musique_N20.pkl:score
for d in musique hotpotqa 2wikimultihopqa; do
  for b in static_adaptive static_fixed chain_adaptive chain_fixed dense_fixed lac_adaptive; do $PY code/loop_signif.py $d lach_adaptive $b; done
  for b in static_fixed chain_fixed dense_fixed lac_fixed; do $PY code/loop_signif.py $d lach_fixed $b; done
done > logs/loopsig_q20.log 2>&1
$PY code/gen_tables4.py 2> logs/gen_tables.err; tail -1 logs/gen_tables.err
echo FINAL20_DONE
