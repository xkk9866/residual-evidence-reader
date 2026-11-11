#!/bin/bash
# lacc on HotpotQA/2Wiki (same recipe as lacr), then all remaining QA (GPU0, after qa_r9)
cd /data/yaokang/proofset; PY=.venv/bin/python
export HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
st() { echo "$(date) $*" >> logs/q35_status.log; }
while pgrep -f "code/llm_tools.py qamulti" > /dev/null; do sleep 60; done
for d in hotpotqa 2wiki; do
  [ -f done/tr_${d}_lacc ] || { st train $d; $PY code/train3.py --data $d --init ckpt/${d}_jbce/model.pt --epochs 1 --max_train 20000 --objective closure --out ckpt/${d}_lacc > logs/${d}_lacc.log 2>&1 && touch done/tr_${d}_lacc; }
  [ -f done/inf_${d}_lacc ] || { st inf $d; $PY code/train3.py --objective closure --ckpt ckpt/${d}_lacc/model.pt --out ckpt/${d}_lacc --infer data/proc/${d}_dev.jsonl >> logs/q35.log 2>&1 && touch done/inf_${d}_lacc; }
  $PY code/eval_select.py ${d}_dev lacc=ckpt/${d}_lacc/pred_${d}_dev.pkl:order >> logs/q35.log 2>&1
done
$PY code/eval_select.py musique_dev lacc=ckpt/mq_lacc/pred_musique_dev.pkl:order >> logs/q35.log 2>&1
for d in musique hotpotqa 2wikimultihopqa; do $PY code/eval_select.py openpool_${d}_N20 lacc=ckpt/mq_lacc/pred_openpool_${d}_N20.pkl:order >> logs/q35.log 2>&1; done
st qa
export RTAG=qwen3_8b
[ -f done/qa35a ] || { $PY code/llm_tools.py qamulti "musique_dev|lacc:k2,k4,t512" "openpool_musique_N20|lacc:k4" "openpool_hotpotqa_N20|lacc:k4" "openpool_2wikimultihopqa_N20|lacc:k4" > logs/qa35a.log 2>&1 && touch done/qa35a; }
[ -f done/qa35b ] || { QA_N=2000 $PY code/llm_tools.py qamulti "hotpotqa_dev|lacc:k2,t384" "2wiki_dev|lacc:k2,t384" > logs/qa35b.log 2>&1 && touch done/qa35b; }
export RTAG=qwen3_4b READER=$HOME/models/Qwen3-4B-AWQ
[ -f done/qa35c ] || { $PY code/llm_tools.py qamulti 'musique_dev|lacc:k4' 'loop_musique_lacc_fixed_r2|lacc_fixed:k4' 'loop_hotpotqa_lacc_fixed_r2|lacc_fixed:k4' 'loop_2wikimultihopqa_lacc_fixed_r2|lacc_fixed:k4' 'loop_musique_lacc_adaptive_r2|lacc_adaptive:k4' 'loop_hotpotqa_lacc_adaptive_r2|lacc_adaptive:k4' 'loop_2wikimultihopqa_lacc_adaptive_r2|lacc_adaptive:k4' > logs/qa35c.log 2>&1 && touch done/qa35c; }
st ALL
