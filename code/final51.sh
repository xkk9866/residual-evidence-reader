#!/bin/bash
# round q50 (final): CPU evaluation of the new baselines and of the exact-set analysis, then the tables (gen_tables8.py)
cd /data/yaokang/proofset; PY=.venv/bin/python; E=code/eval_select.py; D=musique_dev; F=musique_full_dev
# locally retrained Beam Retrieval: swap into the 'beam' slot (official predictions kept as beamoff), both protocols, three dev sets
for d in musique hotpotqa 2wiki; do
  [ -f results/scores/${d}_dev_beam_official.pkl ] || cp results/scores/${d}_dev_beam.pkl results/scores/${d}_dev_beam_official.pkl
  [ -f results/scores/${d}_beamL.pkl ] && { cp results/scores/${d}_beamL.pkl results/scores/${d}_dev_beam.pkl; cp results/scores/${d}_beamK.pkl results/scores/${d}_dev_beamK.pkl; }
  $PY $E ${d}_dev beam=results/scores/${d}_dev_beam.pkl:sets beamk=results/scores/${d}_dev_beamK.pkl:sets beamoff=results/scores/${d}_dev_beam_official.pkl:sets
done
[ -f results/scores/musique_beam5L.pkl ] && $PY $E $D beam5=results/scores/musique_beam5L.pkl:sets beam5k=results/scores/musique_beam5K.pkl:sets
# two-sided reader (analysis), leave-one-out release, proof identification (round 12), training-budget controls
SP=""; FS=""; DSL=""
for n in delta:mq_delta2 delta_s43:mq_delta_s43 delta_s44:mq_delta_s44 delta_depth:mq_delta_depth delta_neg2:mq_delta_neg2 lacc_loo:diag_lacc_loo delta_loo:diag_delta2_loo \
         delta_id:mq_delta_id lacc_id:mq_lacc_id lacc_idb:mq_lacc_idb lacc_e3:mq_lacc_e3 delta2_idrule:mq_delta2_idrule lacc_idrule:mq_lacc_idrule; do
  nm=${n%%:*}; d=ckpt/${n##*:}; [ -f $d/pred_$D.pkl ] || continue
  SP="$SP $nm=$d/pred_$D.pkl:order"; DSL="$DSL $nm=$d"
  [ -f $d/pred_$F.pkl ] && FS="$FS $nm=$d/pred_$F.pkl:order"
done
[ -f ckpt/mq_jbce5/pred_$D.pkl ] && SP="$SP jbce5=ckpt/mq_jbce5/pred_$D.pkl:score"
[ -n "$SP" ] && $PY $E $D $SP
[ -n "$FS" ] && $PY code/ans_eval.py $FS
[ -n "$DSL" ] && $PY code/delta_stats.py $DSL > /dev/null
$PY code/halt_stats2.py > /dev/null
$PY code/calib.py lacc=ckpt/mq_lacc/pred_$D.pkl lacc_s43=ckpt/mq_lacc_s43/pred_$D.pkl lacc_s44=ckpt/mq_lacc_s44/pred_$D.pkl lach42=ckpt/mq_lacr_ev/pred_$D.pkl lach43=ckpt/mq_lacr_s43/pred_$D.pkl lach44=ckpt/mq_lacr_s44/pred_$D.pkl lace42=ckpt/mq_lace_s42/pred_$D.pkl lace43=ckpt/mq_lace_s43/pred_$D.pkl lace44=ckpt/mq_lace_s44/pred_$D.pkl laccr=ckpt/mq_laccr/pred_$D.pkl $(for n in delta:mq_delta2 delta_s43:mq_delta_s43 delta_depth:mq_delta_depth; do d=ckpt/${n##*:}; [ -f $d/pred_$D.pkl ] && echo ${n%%:*}=$d/pred_$D.pkl; done)
BMS="lacc=ckpt/mq_lacc/pred_$D.pkl lach=ckpt/mq_lacr/pred_$D.pkl jbce3=ckpt/mq_jbce3/pred_$D.pkl jbce4=ckpt/mq_jbce4/pred_$D.pkl chain=ckpt/mq_chain/pred_$D.pkl lacc_s43=ckpt/mq_lacc_s43/pred_$D.pkl lacc_s44=ckpt/mq_lacc_s44/pred_$D.pkl"
for n in delta:mq_delta2 delta_s43:mq_delta_s43 delta_s44:mq_delta_s44 delta_depth:mq_delta_depth delta_neg2:mq_delta_neg2 lacc_loo:diag_lacc_loo delta_loo:diag_delta2_loo \
         delta_id:mq_delta_id lacc_id:mq_lacc_id lacc_e3:mq_lacc_e3 jbce5:mq_jbce5 delta2_idrule:mq_delta2_idrule lacc_idrule:mq_lacc_idrule; do d=ckpt/${n##*:}; [ -f $d/pred_$D.pkl ] && BMS="$BMS ${n%%:*}=$d/pred_$D.pkl"; done
$PY code/beam_matched.py $BMS > /dev/null
for n in delta_id lacc_id lacc_e3 jbce5; do [ -f results/sel/${D}_$n.json ] && $PY code/signif.py $D $n lacc k2,k3,k4,k5,t512,t768; done >> logs/signif_q51.log 2>&1
for b in jbce3 jbce4 chain lach beam beamk beamoff; do $PY code/signif.py $D lacc $b k2,k3,k4,k5,t384,t512,t768,t1024; done > logs/signif_q51.log 2>&1
for d in hotpotqa 2wiki; do for b in beam beamk beamoff jbce; do $PY code/signif.py ${d}_dev lacc $b k2,k3,k4,t384,t512; done; done >> logs/signif_q51.log 2>&1
$PY code/gen_tables8.py 2> logs/gen8.err; tail -n 2 logs/gen8.err
echo FINAL51_DONE
