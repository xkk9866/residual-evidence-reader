#!/bin/bash
# round q50: every CPU evaluation of the two-sided residual reader after the GPU jobs, then the tables
cd /data/yaokang/proofset; PY=.venv/bin/python; E=code/eval_select.py; D=musique_dev; F=musique_full_dev
SP=""; FS=""; DS=""; CS=""
for n in delta delta_s43 delta_s44 delta_lat delta_rank delta_depth delta_wl delta_plain delta_Anone delta_Aundo; do
  d=ckpt/mq_$n; [ -f $d/pred_$D.pkl ] || continue
  SP="$SP $n=$d/pred_$D.pkl:order"; DS="$DS $n=$d"; CS="$CS $n=$d/pred_$D.pkl"
  [ -f $d/pred_$F.pkl ] && FS="$FS $n=$d/pred_$F.pkl:order"
done
for W in 2 3 4 6; do [ -f ckpt/mq_delta_W$W/pred_$D.pkl ] && { SP="$SP delta_W$W=ckpt/mq_delta_W$W/pred_$D.pkl:order"; DS="$DS delta_W$W=ckpt/mq_delta_W$W"; }; done
[ -f ckpt/mq_delta_L320/pred_$D.pkl ] && { SP="$SP delta_L320=ckpt/mq_delta_L320/pred_$D.pkl:order"; DS="$DS delta_L320=ckpt/mq_delta_L320"; }
$PY $E $D $SP
$PY code/ans_eval.py $FS
$PY code/delta_stats.py $DS
$PY code/halt_stats2.py > /dev/null
$PY code/calib.py lacc=ckpt/mq_lacc/pred_$D.pkl lacc_s43=ckpt/mq_lacc_s43/pred_$D.pkl lacc_s44=ckpt/mq_lacc_s44/pred_$D.pkl lach42=ckpt/mq_lacr_ev/pred_$D.pkl lach43=ckpt/mq_lacr_s43/pred_$D.pkl lach44=ckpt/mq_lacr_s44/pred_$D.pkl lace42=ckpt/mq_lace_s42/pred_$D.pkl lace43=ckpt/mq_lace_s43/pred_$D.pkl lace44=ckpt/mq_lace_s44/pred_$D.pkl laccr=ckpt/mq_laccr/pred_$D.pkl $CS
BMS=""
for n in lacc lacr_ev:lach jbce3 jbce4 chain lacc_s43 lacc_s44 lacr_s43:lach_s43 lacr_s44:lach_s44 lacp lact lace_s42:lace laccr abl_ce1 abl_noroll abl_noworld abl_mark abl_prefix abl_rbce lacc_T0 lacc_T1 lacc_T2 lacc_T3 lacc_T4; do
  src=${n%%:*}; tag=${n##*:}; p=ckpt/mq_$src/pred_$D.pkl; [ -f $p ] || p=ckpt/$src/pred_$D.pkl; [ -f $p ] && BMS="$BMS $tag=$p"
done
for n in delta_depth delta_wl delta_plain delta_Anone delta_Aundo; do [ -f ckpt/mq_$n/pred_$D.pkl ] && BMS="$BMS $n=ckpt/mq_$n/pred_$D.pkl"; done
for W in 2 3 4 6; do [ -f ckpt/mq_delta_W$W/pred_$D.pkl ] && BMS="$BMS delta_W$W=ckpt/mq_delta_W$W/pred_$D.pkl"; done
$PY code/beam_matched.py delta=ckpt/mq_delta/pred_$D.pkl delta_s43=ckpt/mq_delta_s43/pred_$D.pkl delta_s44=ckpt/mq_delta_s44/pred_$D.pkl delta_lat=ckpt/mq_delta_lat/pred_$D.pkl delta_rank=ckpt/mq_delta_rank/pred_$D.pkl $BMS > /dev/null
for b in lacc jbce3 jbce4 chain lach beam beamk beamoff; do $PY code/signif.py $D delta $b k2,k3,k4,k5,t384,t512,t768,t1024; done > logs/signif_q50.log 2>&1
for s in 43 44; do $PY code/signif.py $D delta_s$s lacc_s$s k2,k3,k4,k5,t768 >> logs/signif_q50.log 2>&1; done
for d in hotpotqa 2wiki; do
  [ -f ckpt/${d}_delta/pred_${d}_dev.pkl ] && $PY $E ${d}_dev delta=ckpt/${d}_delta/pred_${d}_dev.pkl:order
  for b in lacc beam beamk beamoff jbce; do $PY code/signif.py ${d}_dev delta $b k2,k3,k4,t384,t512; done >> logs/signif_q50.log 2>&1
done
for d in musique hotpotqa 2wikimultihopqa; do
  P=openpool_${d}_N20; [ -f ckpt/mq_delta/pred_$P.pkl ] && $PY $E $P delta=ckpt/mq_delta/pred_$P.pkl:order
  P=openpool_${d}_N40; [ -f ckpt/mq_delta/pred_$P.pkl ] && $PY $E $P delta=ckpt/mq_delta/pred_$P.pkl:order
done
[ -f ckpt/mq_delta_L320/pred_openpool_musique_N20.pkl ] && $PY $E openpool_musique_N20 delta_L320=ckpt/mq_delta_L320/pred_openpool_musique_N20.pkl:order
for d in musique hotpotqa 2wikimultihopqa; do
  bash code/loop_eval.sh $d > logs/loopeval50_$d.txt 2>&1
  for b in static_fixed chain_fixed dense_fixed lacc_fixed; do $PY code/loop_signif.py $d delta_fixed $b; done
  for b in static_adaptive static_fixed lacc_adaptive delta-anscls_adaptive static-delta_adaptive; do $PY code/loop_signif.py $d delta_adaptive $b; done
done > logs/loopsig_q50.log 2>&1
$PY code/gen_tables7.py 2> logs/gen7.err; tail -n 2 logs/gen7.err
echo FINAL50_DONE
