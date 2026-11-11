#!/bin/bash
# GPU-6 ONLY sequential queue for LACUNA (residual-proof reader, evidence-as-query layout).
# Lattice curriculum: stage 1 = bottom of the lattice (S = empty; static joint reader, train2 --head bce),
#                     stage 2 = the full residual lattice (train3 --init stage1).  Control: stage 1 + 1 more static epoch.
cd /data/yaokang/proofset
export HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=6 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python
ST=logs/q5_status.log
run() { echo "[$(date)] START $*" >> $ST; "$@"; echo "[$(date)] END rc=$? $*" >> $ST; }
D=musique_dev
INF="data/proc/musique_dev.jsonl,data/proc/musique_full_dev.jsonl"
# wait for the running stage-2 job (ckpt/mq_lacq)
while pgrep -f "^.venv/bin/python code/train3.py --data musique --init" >/dev/null; do sleep 30; done
# 1) MuSiQue main
run $PY code/train3.py --ckpt ckpt/mq_lacq/model.pt --out ckpt/mq_lacq --infer $INF > logs/mq_lacq_infer.log 2>&1
run $PY code/decode3.py --ckpt ckpt/mq_lacq/model.pt --out ckpt/mq_lacq --infer data/proc/musique_dev.jsonl --width 3 > logs/mq_lacq_pv.log 2>&1
run $PY code/eval_select.py $D lac=ckpt/mq_lacq/pred_$D.pkl:order lacpv=ckpt/mq_lacq/pred_${D}_pv3.pkl:pv > logs/eval_mq5.log 2>&1
run $PY code/train2.py --data musique --head bce --ckpt ckpt/mq_jbce/model.pt --out ckpt/mq_jbce3 --epochs 1 > logs/mq_jbce3.log 2>&1
run $PY code/train2.py --head bce --ckpt ckpt/mq_jbce3/model.pt --out ckpt/mq_jbce3 --infer $INF > logs/mq_jbce3_infer.log 2>&1
run $PY code/eval_select.py $D jbce3=ckpt/mq_jbce3/pred_$D.pkl:score prove2=ckpt/mq_v2/pred_$D.pkl:circuit jbce=ckpt/mq_jbce/pred_$D.pkl:score point=ckpt/musique_point/pred_$D.pkl:score beam=results/scores/${D}_beam.pkl:sets >> logs/eval_mq5.log 2>&1
run $PY code/ans_eval.py lac=ckpt/mq_lacq/pred_musique_full_dev.pkl:order prove2=ckpt/mq_v2/pred_musique_full_dev.pkl:circuit jbce3=ckpt/mq_jbce3/pred_musique_full_dev.pkl:score jbce=ckpt/mq_jbce/pred_musique_full_dev.pkl:score > logs/ans_mq.log 2>&1
run $PY code/diag3.py $D ckpt/mq_lacq/pred_$D.pkl ckpt/mq_jbce3/pred_$D.pkl > logs/diag_mq.log 2>&1
# 2) LLM (int4 Qwen3-8B-AWQ): SetR-style baseline + QA on MuSiQue
run $PY code/llm_tools.py setr $D > logs/setr_mq.log 2>&1
run $PY code/eval_select.py $D setr=results/scores/${D}_setr_qwen3_8b.pkl:sets >> logs/eval_mq5.log 2>&1
run $PY code/llm_tools.py qa $D lac:k2,k4,t512 lacpv:k4 jbce3:k2,k4,t512 point:k4,t512 beam:k2,k4,t512 prove2:k4,t512 setr:k4,t512 oracle:k4 fullpool:k4 noctx:k4 > logs/qa_mq5.log 2>&1
# 3) HotpotQA / 2Wiki (20k questions): stage 1 static (1 epoch) -> stage 2 residual (1 epoch)
for ds in hotpotqa 2wiki; do
  run $PY code/train2.py --data $ds --head bce --out ckpt/${ds}_jbce --epochs 1 --max_train 20000 > logs/${ds}_jbce.log 2>&1
  run $PY code/train3.py --data $ds --init ckpt/${ds}_jbce/model.pt --out ckpt/${ds}_lacq --epochs 1 --max_train 20000 > logs/${ds}_lacq.log 2>&1
  run $PY code/train2.py --head bce --ckpt ckpt/${ds}_jbce/model.pt --out ckpt/${ds}_jbce --infer data/proc/${ds}_dev.jsonl > logs/${ds}_jbce_infer.log 2>&1
  run $PY code/train3.py --ckpt ckpt/${ds}_lacq/model.pt --out ckpt/${ds}_lacq --infer data/proc/${ds}_dev.jsonl --steps 4 > logs/${ds}_lacq_infer.log 2>&1
  run $PY code/eval_select.py ${ds}_dev lac=ckpt/${ds}_lacq/pred_${ds}_dev.pkl:order jbce=ckpt/${ds}_jbce/pred_${ds}_dev.pkl:score prove2=ckpt/${ds}_v2/pred_${ds}_dev.pkl:circuit beam=results/scores/${ds}_dev_beam.pkl:sets > logs/eval_${ds}5.log 2>&1
done
export QA_N=2000
for ds in hotpotqa 2wiki; do
  run $PY code/llm_tools.py qa ${ds}_dev lac:k2,t384 jbce:k2,t384 beam:k2,t384 oracle:k2 > logs/qa_${ds}5.log 2>&1
done
unset QA_N
# 4) ablations on MuSiQue (stage-2 variants, 1 epoch each from the same stage-1 reader)
for v in "mark --layout mark" "prefix --states prefix --no_roll" "noroll --no_roll" "noworld --no_world"; do
  set -- $v; a=mq_lacq_$1; shift
  run $PY code/train3.py --data musique --init ckpt/mq_jbce/model.pt --out ckpt/$a --epochs 1 "$@" > logs/$a.log 2>&1
  lay=""; [ $a = mq_lacq_mark ] && lay="--layout mark"
  run $PY code/train3.py $lay --ckpt ckpt/$a/model.pt --out ckpt/$a --infer $INF > logs/${a}_infer.log 2>&1
  run $PY code/eval_select.py $D $a=ckpt/$a/pred_$D.pkl:order >> logs/eval_abl5.log 2>&1
done
run $PY code/ans_eval.py mq_lacq_mark=ckpt/mq_lacq_mark/pred_musique_full_dev.pkl:order mq_lacq_prefix=ckpt/mq_lacq_prefix/pred_musique_full_dev.pkl:order mq_lacq_noroll=ckpt/mq_lacq_noroll/pred_musique_full_dev.pkl:order mq_lacq_noworld=ckpt/mq_lacq_noworld/pred_musique_full_dev.pkl:order > logs/ans_abl5.log 2>&1
echo "[$(date)] ALL DONE" >> $ST
