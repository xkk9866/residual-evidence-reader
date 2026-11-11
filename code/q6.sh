#!/bin/bash
# runs after q5.sh: decision-set QA on HotpotQA/2Wiki, significance tests, tables.  GPU 6 only.
cd /data/yaokang/proofset
export HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=6 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python
ST=logs/q6_status.log
run() { echo "[$(date)] START $*" >> $ST; "$@"; echo "[$(date)] END rc=$? $*" >> $ST; }
while ! grep -q "ALL DONE" logs/q5_status.log; do sleep 60; done
run $PY code/llm_tools.py qa musique_dev lac:dec jbce3:dec prove2:dec point:dec beam:k5 > logs/qa_mq_dec.log 2>&1
run $PY code/signif.py musique_dev lac jbce3 > logs/signif_mq.log 2>&1
run $PY code/signif.py musique_dev lac beam >> logs/signif_mq.log 2>&1
export QA_N=2000
for ds in hotpotqa 2wiki; do
  run $PY code/llm_tools.py qa ${ds}_dev lac:dec jbce:dec beam:k5 > logs/qa_${ds}_dec.log 2>&1
  run $PY code/signif.py ${ds}_dev lac jbce k2,k3,k4,t256,t384 > logs/signif_${ds}.log 2>&1
  run $PY code/signif.py ${ds}_dev lac beam k2,k3,k4,t256,t384 >> logs/signif_${ds}.log 2>&1
done
unset QA_N
for a in mq_lacq_mark mq_lacq_prefix mq_lacq_noroll mq_lacq_noworld; do
  run $PY code/signif.py musique_dev lac $a k2,k3,k4,k5,t512 >> logs/signif_abl.log 2>&1
done
run $PY code/gen_tables.py > logs/gen_tables.log 2>&1
# seeds (stage 2 and the static control), MuSiQue
for s in 43 44; do
  run $PY code/train3.py --data musique --init ckpt/mq_jbce/model.pt --out ckpt/mq_lacq_s$s --epochs 1 --seed $s > logs/mq_lacq_s$s.log 2>&1
  run $PY code/train3.py --ckpt ckpt/mq_lacq_s$s/model.pt --out ckpt/mq_lacq_s$s --infer data/proc/musique_dev.jsonl,data/proc/musique_full_dev.jsonl > logs/mq_lacq_s${s}_infer.log 2>&1
  run $PY code/train2.py --data musique --head bce --ckpt ckpt/mq_jbce/model.pt --out ckpt/mq_jbce3_s$s --epochs 1 --seed $s > logs/mq_jbce3_s$s.log 2>&1
  run $PY code/train2.py --head bce --ckpt ckpt/mq_jbce3_s$s/model.pt --out ckpt/mq_jbce3_s$s --infer data/proc/musique_dev.jsonl,data/proc/musique_full_dev.jsonl > logs/mq_jbce3_s${s}_infer.log 2>&1
  run $PY code/eval_select.py musique_dev lac_s$s=ckpt/mq_lacq_s$s/pred_musique_dev.pkl:order jbce3_s$s=ckpt/mq_jbce3_s$s/pred_musique_dev.pkl:score >> logs/eval_seeds.log 2>&1
  run $PY code/ans_eval.py lac_s$s=ckpt/mq_lacq_s$s/pred_musique_full_dev.pkl:order jbce3_s$s=ckpt/mq_jbce3_s$s/pred_musique_full_dev.pkl:score >> logs/ans_seeds.log 2>&1
done
run $PY code/gen_tables.py > logs/gen_tables.log 2>&1
echo "[$(date)] ALL DONE" >> $ST
