#!/bin/bash
# loop_eval.sh DS : budgeted CER of every loop at every round (sets for QA are written to results/sel)
cd /data/yaokang/proofset; PY=.venv/bin/python; DS=$1
declare -A SP=([lac]="ckpt/mq_lact:order" [lach]="ckpt/mq_lacr_ev:order" [lacc]="ckpt/mq_lacc:order" [delta]="ckpt/mq_delta:order" [static]="ckpt/mq_jbce3:score" [chain]="ckpt/mq_chain:order" [dense]="dense:score")
for f in data/proc/loop_${DS}_*_r2.jsonl; do
  T=$(basename $f _r2.jsonl); M=$(echo $T | sed "s/loop_${DS}_//; s/_\(fixed\|adaptive\)$//")
  for R in 0 1 2; do
    P=$([ $R = 0 ] && echo openpool_${DS}_N20 || echo ${T}_r$R)
    K=${M%%-*}; dir=${SP[$K]%%:*}; typ=${SP[$K]##*:}
    : ${dir}
    if [ $M = dense ]; then
      $PY -c "import json,pickle,numpy as np; R=[json.loads(l) for l in open('data/proc/$P.jsonl')]; pickle.dump([{'rho':-np.arange(len(r['docs']),dtype=float)} for r in R],open('results/scores/${P}_dense.pkl','wb'))"
      path=results/scores/${P}_dense.pkl
    else path=$dir/pred_$P.pkl; fi
    $PY code/eval_select.py $P ${M}_${T##*_}=$path:$typ | sed "s/^/$T r$R /"
  done
done
