#!/bin/bash
# GPU-6 ONLY sequential queue for PROVE v2 (joint reader + transport witnesses + proof likelihood)
cd /data/yaokang/proofset
export HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=6 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python
run() { echo "[$(date)] START $*" >> logs/q3_status.log; "$@"; echo "[$(date)] END rc=$? $*" >> logs/q3_status.log; }
# 1) MuSiQue main + same-reader BCE ablation
run $PY code/train2.py --data musique --loss proof --out ckpt/mq_v2 --epochs 2 > logs/mq_v2.log 2>&1
run $PY code/train2.py --ckpt ckpt/mq_v2/model.pt --out ckpt/mq_v2 --infer data/proc/musique_dev.jsonl,data/proc/musique_full_dev.jsonl > logs/mq_v2_infer.log 2>&1
run $PY code/train2.py --data musique --head bce --out ckpt/mq_jbce --epochs 2 > logs/mq_jbce.log 2>&1
run $PY code/train2.py --head bce --ckpt ckpt/mq_jbce/model.pt --out ckpt/mq_jbce --infer data/proc/musique_dev.jsonl,data/proc/musique_full_dev.jsonl > logs/mq_jbce_infer.log 2>&1
# 2) HotpotQA / 2Wiki
for ds in hotpotqa 2wiki; do
  run $PY code/train2.py --data $ds --loss proof --out ckpt/${ds}_v2 --epochs 1 --max_train 30000 > logs/${ds}_v2.log 2>&1
  run $PY code/train2.py --ckpt ckpt/${ds}_v2/model.pt --out ckpt/${ds}_v2 --infer data/proc/${ds}_dev.jsonl > logs/${ds}_v2_infer.log 2>&1
done
# 3) ablations on MuSiQue (1 epoch each)
run $PY code/train2.py --data musique --loss proof --no_transport --out ckpt/mq_v2_notr --epochs 1 > logs/mq_v2_notr.log 2>&1
run $PY code/train2.py --data musique --loss proof --no_world --out ckpt/mq_v2_noworld --epochs 1 > logs/mq_v2_noworld.log 2>&1
for a in mq_v2_notr mq_v2_noworld; do
  extra=""; [ $a = mq_v2_notr ] && extra="--no_transport"
  run $PY code/train2.py $extra --ckpt ckpt/$a/model.pt --out ckpt/$a --infer data/proc/musique_dev.jsonl,data/proc/musique_full_dev.jsonl > logs/${a}_infer.log 2>&1
done
# 4) selection eval + LLM (int4 Qwen2.5-7B-AWQ) QA / SetR baseline
D=musique_dev
run $PY code/eval_select.py $D prove2=ckpt/mq_v2/pred_$D.pkl:circuit jbce=ckpt/mq_jbce/pred_$D.pkl:score point=ckpt/musique_point/pred_$D.pkl:score beam=results/scores/${D}_beam.pkl:sets > logs/eval_mq.log 2>&1
run $PY code/llm_tools.py setr $D > logs/setr_mq.log 2>&1
run $PY code/llm_tools.py qa $D prove2:k4,t512 jbce:k4,t512 point:k4,t512 beam:k4,t512 oracle:k4 fullpool:k4 noctx:k4 > logs/qa_mq.log 2>&1
echo "[$(date)] ALL DONE" >> logs/q3_status.log
