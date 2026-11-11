#!/bin/bash
# FINAL inference / evaluation queue, runs in PARALLEL with q7_train.sh on GPU 6 (inference needs ~3-6 GB).
# Waits for done/<name> markers.  At the end (after all training) runs the vLLM jobs, which need the GPU alone.
cd /data/yaokang/proofset
export HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=6 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python
ST=logs/q8_status.log
run() { echo "[$(date)] START $*" >> $ST; "$@"; echo "[$(date)] END rc=$? $*" >> $ST; }
waitfor() { while [ ! -f done/$1 ]; do sleep 60; done; }
D=musique_dev; F=musique_full_dev
INF="data/proc/$D.jsonl,data/proc/$F.jsonl"
greedy() { run $PY code/train3.py --ckpt ckpt/$1/model.pt --out ckpt/$1 --infer $2 > logs/$1_infer.log 2>&1; }
ns() { run $PY code/ns_decode.py --ckpt ckpt/$1/model.pt --out ckpt/$1 --infer $2 --width $3 > logs/$1_ns$3.log 2>&1; }
# 0) zero-shot Qwen3-Reranker-4B (int8) on all dev sets + stage-1 HotpotQA reader (no training needed)
for ds in musique_dev hotpotqa_dev 2wiki_dev; do run $PY code/q3rr.py $ds > logs/q3rr_$ds.log 2>&1; done
run $PY code/train2.py --head bce --ckpt ckpt/hotpotqa_jbce/model.pt --out ckpt/hotpotqa_jbce --infer data/proc/hotpotqa_dev.jsonl > logs/hotpotqa_jbce_infer.log 2>&1
# 1) MuSiQue main
waitfor mq_lacr
greedy mq_lacr $INF
ns mq_lacr data/proc/$D.jsonl 3
ns mq_lacr data/proc/$F.jsonl 1
run $PY code/eval_select.py $D lac=ckpt/mq_lacr/pred_${D}_ns3.pkl:ns lacg=ckpt/mq_lacr/pred_$D.pkl:order q3rr=results/scores/${D}_q3rr.pkl:score \
  jbce3=ckpt/mq_jbce3/pred_$D.pkl:score jbce=ckpt/mq_jbce/pred_$D.pkl:score prove2=ckpt/mq_v2/pred_$D.pkl:circuit point=ckpt/musique_point/pred_$D.pkl:score \
  beam=results/scores/${D}_beam.pkl:sets setr=results/scores/${D}_setr_qwen3_8b.pkl:sets rerank=results/scores/${D}_rerank.pkl:score \
  bm25=results/scores/${D}_bm25.pkl:score dense=results/scores/${D}_dense.pkl:score > logs/eval_final_mq.log 2>&1
waitfor mq_chain
run $PY code/chain_ce.py --ckpt ckpt/mq_chain/model.pt --out ckpt/mq_chain --infer data/proc/$D.jsonl > logs/mq_chain_infer.log 2>&1
run $PY code/eval_select.py $D chain=ckpt/mq_chain/pred_$D.pkl:order >> logs/eval_final_mq.log 2>&1
waitfor mq_anscls
run $PY code/train3.py --objective anscls --ckpt ckpt/mq_anscls/model.pt --out ckpt/mq_anscls --infer data/proc/$F.jsonl > logs/mq_anscls_infer.log 2>&1
# 2) HotpotQA / 2Wiki
for ds in hotpotqa 2wiki; do
  waitfor ${ds}_lacr
  [ $ds = 2wiki ] && run $PY code/train2.py --head bce --ckpt ckpt/2wiki_jbce/model.pt --out ckpt/2wiki_jbce --infer data/proc/2wiki_dev.jsonl > logs/2wiki_jbce_infer.log 2>&1
  run $PY code/train3.py --ckpt ckpt/${ds}_lacr/model.pt --out ckpt/${ds}_lacr --infer data/proc/${ds}_dev.jsonl --steps 4 > logs/${ds}_lacr_infer.log 2>&1
  run $PY code/ns_decode.py --ckpt ckpt/${ds}_lacr/model.pt --out ckpt/${ds}_lacr --infer data/proc/${ds}_dev.jsonl --width 1 --steps 4 > logs/${ds}_lacr_ns1.log 2>&1
  run $PY code/eval_select.py ${ds}_dev lac=ckpt/${ds}_lacr/pred_${ds}_dev_ns1.pkl:ns lacg=ckpt/${ds}_lacr/pred_${ds}_dev.pkl:order jbce=ckpt/${ds}_jbce/pred_${ds}_dev.pkl:score \
    prove2=ckpt/${ds}_v2/pred_${ds}_dev.pkl:circuit beam=results/scores/${ds}_dev_beam.pkl:sets q3rr=results/scores/${ds}_dev_q3rr.pkl:score \
    rerank=results/scores/${ds}_dev_rerank.pkl:score bm25=results/scores/${ds}_dev_bm25.pkl:score > logs/eval_final_${ds}.log 2>&1
done
# 3) ablations + seeds (greedy completion, same decoding for every variant)
for a in abl_mark abl_prefix abl_noroll abl_noworld mq_lacr_s43; do
  waitfor $a
  lay=""; [ $a = abl_mark ] && lay="--layout mark"
  run $PY code/train3.py $lay --ckpt ckpt/$a/model.pt --out ckpt/$a --infer $INF > logs/${a}_infer.log 2>&1
  run $PY code/eval_select.py $D $a=ckpt/$a/pred_$D.pkl:order >> logs/eval_final_abl.log 2>&1
done
waitfor mq_jbce3_s43
run $PY code/train2.py --head bce --ckpt ckpt/mq_jbce3_s43/model.pt --out ckpt/mq_jbce3_s43 --infer $INF > logs/mq_jbce3_s43_infer.log 2>&1
run $PY code/eval_select.py $D jbce3_s43=ckpt/mq_jbce3_s43/pred_$D.pkl:score >> logs/eval_final_abl.log 2>&1
run $PY code/ans_eval.py lac=ckpt/mq_lacr/pred_${F}_ns1.pkl:ns lacg=ckpt/mq_lacr/pred_$F.pkl:order anscls=ckpt/mq_anscls/pred_$F.pkl:cls \
  prove2=ckpt/mq_v2/pred_$F.pkl:circuit jbce3=ckpt/mq_jbce3/pred_$F.pkl:score abl_mark=ckpt/abl_mark/pred_$F.pkl:order \
  abl_prefix=ckpt/abl_prefix/pred_$F.pkl:order abl_noroll=ckpt/abl_noroll/pred_$F.pkl:order abl_noworld=ckpt/abl_noworld/pred_$F.pkl:order \
  mq_lacr_s43=ckpt/mq_lacr_s43/pred_$F.pkl:order jbce3_s43=ckpt/mq_jbce3_s43/pred_$F.pkl:score > logs/ans_final.log 2>&1
# 4) LLM jobs (vLLM needs the whole GPU): wait for all training
waitfor TRAIN_ALL
run $PY code/llm_tools.py rank $D > logs/rank_mq.log 2>&1
run $PY code/eval_select.py $D rankgpt=results/scores/${D}_rankgpt_qwen3_8b.pkl:sets >> logs/eval_final_mq.log 2>&1
run $PY code/llm_tools.py qa $D lac:k2,k4,t512,dec jbce3:k2,k4,t512,dec chain:k2,k4,t512 beam:k2,k4,t512,k5 q3rr:k2,k4,t512 \
  rankgpt:k2,k4,t512 setr:k2,k4,t512 point:k2,k4,t512,dec prove2:k2,k4,t512,dec oracle:k2 fullpool:k2 noctx:k2 > logs/qa_final_mq.log 2>&1
export QA_N=2000
for ds in hotpotqa 2wiki; do
  run $PY code/llm_tools.py qa ${ds}_dev lac:k2,t384,dec jbce:k2,t384,dec beam:k2,t384,k5 q3rr:k2,t384 oracle:k2 > logs/qa_final_${ds}.log 2>&1
done
unset QA_N
# 5) significance + tables
run $PY code/signif.py $D lac jbce3 > logs/signif_final.log 2>&1
for b in beam chain lacg q3rr; do run $PY code/signif.py $D lac $b >> logs/signif_final.log 2>&1; done
for ds in hotpotqa 2wiki; do run $PY code/signif.py ${ds}_dev lac jbce k2,k3,k4,t256,t384,t512 >> logs/signif_final.log 2>&1; run $PY code/signif.py ${ds}_dev lac beam k2,k3,k4,t256,t384,t512 >> logs/signif_final.log 2>&1; done
run $PY code/gen_tables.py > logs/gen_tables.log 2>&1
touch done/ALL
echo "[$(date)] ALL DONE" >> $ST
