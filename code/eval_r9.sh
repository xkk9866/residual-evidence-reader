#!/bin/bash
# round 9 evaluation (CPU): budget CER, answerability, stop, calibration, first errors for the closure models
cd /data/yaokang/proofset; PY=.venv/bin/python
SP=""; FS=""; CS=""; FE=""
for n in lacc lacc_s43 lacc_s44 laccr; do
  d=ckpt/mq_$n; [ -f $d/pred_musique_dev.pkl ] || continue
  SP="$SP $n=$d/pred_musique_dev.pkl:order"; CS="$CS $n=$d/pred_musique_dev.pkl"
  [ -f $d/pred_musique_full_dev.pkl ] && FS="$FS $n=$d/pred_musique_full_dev.pkl:order"
done
$PY code/eval_select.py musique_dev $SP
$PY code/ans_eval.py $FS
$PY code/halt_stats2.py > /dev/null
$PY code/calib.py lach42=ckpt/mq_lacr_ev/pred_musique_dev.pkl lach43=ckpt/mq_lacr_s43/pred_musique_dev.pkl lach44=ckpt/mq_lacr_s44/pred_musique_dev.pkl lace42=ckpt/mq_lace_s42/pred_musique_dev.pkl lace43=ckpt/mq_lace_s43/pred_musique_dev.pkl lace44=ckpt/mq_lace_s44/pred_musique_dev.pkl ce1=ckpt/abl_ce1/pred_musique_dev.pkl lact=ckpt/mq_lact/pred_musique_dev.pkl $CS
D=musique_dev; FE=""
for n in lach lach_cal lact lact_cal abl_noroll abl_ce1 abl_noworld lacc lacc_cal laccr; do
  case $n in lach*) p=ckpt/mq_lacr_ev;; lact*) p=ckpt/mq_lact;; lacc*) p=ckpt/mq_${n%_cal};; *) p=ckpt/$n;; esac
  [ -f $p/pred_$D.pkl ] && FE="$FE $n=$p/pred_$D.pkl@results/sel/${D}_${n}_dec.json"
done
$PY code/first_error.py $FE > logs/first_error.log 2>&1
$PY code/beam_matched.py lacc=ckpt/mq_lacc/pred_$D.pkl lach=ckpt/mq_lacr/pred_$D.pkl lact=ckpt/mq_lact/pred_$D.pkl jbce3=ckpt/mq_jbce3/pred_$D.pkl jbce4=ckpt/mq_jbce4/pred_$D.pkl chain=ckpt/mq_chain/pred_$D.pkl > /dev/null
for b in jbce3 jbce4 chain lach beam; do $PY code/signif.py $D lacc $b k2,k3,k4,k5,t384,t512,t768,t1024; done > logs/signif_r9.log 2>&1
echo EVAL_R9_DONE
