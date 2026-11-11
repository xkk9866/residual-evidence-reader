#!/bin/bash
# FINAL single serial queue (GPU 6 only).  Another user's process occupies ~9.5 GB of GPU 6, so training and
# inference cannot overlap; vLLM runs with GPU_UTIL=0.5.  Status: logs/q9_status.log ; markers: done/
# Final config: LACUNA = train3 (evidence-as-query, residual likelihood, lattice + 1-step learner states,
#               world deletion with distractor fill = pool size kept as in MuSiQue-Full), stage 2 from the stage-1 joint reader;
#               decoding = NS-certified (ns_decode.py; width 3 on MuSiQue dev, width 1 elsewhere).
cd /data/yaokang/proofset
export HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=6 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True GPU_UTIL=0.5
PY=.venv/bin/python
ST=logs/q10_status.log
mkdir -p done
run() { echo "[$(date)] START $*" >> $ST; "$@"; echo "[$(date)] END rc=$? $*" >> $ST; }
skip() { [ -f done/$1 ]; }                       # resume support: finished steps are skipped
mark() { touch done/$1; }
T3="$PY code/train3.py --data musique --init ckpt/mq_jbce/model.pt --epochs 1"
D=musique_dev; F=musique_full_dev
INF="data/proc/$D.jsonl,data/proc/$F.jsonl"
greedy() { run $PY code/train3.py $3 --ckpt ckpt/$1/model.pt --out ckpt/$1 --infer $2 > logs/$1_infer.log 2>&1; }
ns() { run $PY code/ns_decode.py --ckpt ckpt/$1/model.pt --out ckpt/$1 --infer $2 --width $3 > logs/$1_ns$3_$(basename $2 .jsonl).log 2>&1; }

# ---------- 1) MuSiQue main model ----------
skip mq_lacr || { run $T3 --out ckpt/mq_lacr > logs/mq_lacr.log 2>&1; greedy mq_lacr $INF; ns mq_lacr data/proc/$D.jsonl 3; ns mq_lacr data/proc/$F.jsonl 1; mark mq_lacr; }
run $PY code/eval_select.py $D lac=ckpt/mq_lacr/pred_${D}_ns3.pkl:ns lacg=ckpt/mq_lacr/pred_$D.pkl:order \
  jbce3=ckpt/mq_jbce3/pred_$D.pkl:score jbce=ckpt/mq_jbce/pred_$D.pkl:score prove2=ckpt/mq_v2/pred_$D.pkl:circuit point=ckpt/musique_point/pred_$D.pkl:score \
  beam=results/scores/${D}_beam.pkl:sets setr=results/scores/${D}_setr_qwen3_8b.pkl:sets rerank=results/scores/${D}_rerank.pkl:score \
  bm25=results/scores/${D}_bm25.pkl:score dense=results/scores/${D}_dense.pkl:score > logs/eval_final_mq.log 2>&1
# ---------- 2) same-backbone chain retriever (Beam-style) ----------
skip mq_chain || { run $PY code/chain_ce.py --data musique --init ckpt/musique_point/model.pt --out ckpt/mq_chain --epochs 1 > logs/mq_chain.log 2>&1
  run $PY code/chain_ce.py --ckpt ckpt/mq_chain/model.pt --out ckpt/mq_chain --infer data/proc/$D.jsonl > logs/mq_chain_infer.log 2>&1; mark mq_chain; }
run $PY code/eval_select.py $D chain=ckpt/mq_chain/pred_$D.pkl:order >> logs/eval_final_mq.log 2>&1
# ---------- 3) same-backbone answerability classifier ----------
skip mq_anscls || { run $T3 --objective anscls --out ckpt/mq_anscls > logs/mq_anscls.log 2>&1
  run $PY code/train3.py --objective anscls --ckpt ckpt/mq_anscls/model.pt --out ckpt/mq_anscls --infer data/proc/$F.jsonl > logs/mq_anscls_infer.log 2>&1; mark mq_anscls; }
# ---------- 4) HotpotQA / 2Wiki ----------
skip hotpotqa_lacr || { run $PY code/train2.py --head bce --ckpt ckpt/hotpotqa_jbce/model.pt --out ckpt/hotpotqa_jbce --infer data/proc/hotpotqa_dev.jsonl > logs/hotpotqa_jbce_infer.log 2>&1
  run $PY code/train3.py --data hotpotqa --init ckpt/hotpotqa_jbce/model.pt --out ckpt/hotpotqa_lacr --epochs 1 --max_train 20000 > logs/hotpotqa_lacr.log 2>&1; mark hotpotqa_lacr; }
while pgrep -f "^.venv/bin/python code/train2.py --data 2wiki" >/dev/null; do sleep 30; done
skip 2wiki_lacr || { [ -f ckpt/2wiki_jbce/model.pt ] && grep -q "^dev 2500" logs/2wiki_jbce.log || run $PY code/train2.py --data 2wiki --head bce --out ckpt/2wiki_jbce --epochs 1 --max_train 20000 > logs/2wiki_jbce.log 2>&1
  run $PY code/train2.py --head bce --ckpt ckpt/2wiki_jbce/model.pt --out ckpt/2wiki_jbce --infer data/proc/2wiki_dev.jsonl > logs/2wiki_jbce_infer.log 2>&1
  run $PY code/train3.py --data 2wiki --init ckpt/2wiki_jbce/model.pt --out ckpt/2wiki_lacr --epochs 1 --max_train 20000 > logs/2wiki_lacr.log 2>&1; mark 2wiki_lacr; }
for ds in hotpotqa 2wiki; do
  skip ${ds}_inf || { run $PY code/train3.py --ckpt ckpt/${ds}_lacr/model.pt --out ckpt/${ds}_lacr --infer data/proc/${ds}_dev.jsonl --steps 4 > logs/${ds}_lacr_infer.log 2>&1
    mark ${ds}_inf; }
done
# ---------- 5) ablations + second seed (greedy completion, same decoding for every variant) ----------
for v in "abl_mark --layout mark" "abl_prefix --states prefix --no_roll" "abl_noroll --no_roll" "abl_noworld --no_world" "mq_lacr_s43 --seed 43"; do
  set -- $v; a=$1; shift
  skip $a || { run $T3 --out ckpt/$a "$@" > logs/$a.log 2>&1
    lay=""; [ $a = abl_mark ] && lay="--layout mark"
    greedy $a $INF "$lay"; mark $a; }
  run $PY code/eval_select.py $D $a=ckpt/$a/pred_$D.pkl:order >> logs/eval_final_abl.log 2>&1
done
skip mq_jbce3_s43 || { run $PY code/train2.py --data musique --head bce --ckpt ckpt/mq_jbce/model.pt --out ckpt/mq_jbce3_s43 --epochs 1 --seed 43 > logs/mq_jbce3_s43.log 2>&1
  run $PY code/train2.py --head bce --ckpt ckpt/mq_jbce3_s43/model.pt --out ckpt/mq_jbce3_s43 --infer $INF > logs/mq_jbce3_s43_infer.log 2>&1; mark mq_jbce3_s43; }
run $PY code/eval_select.py $D jbce3_s43=ckpt/mq_jbce3_s43/pred_$D.pkl:score >> logs/eval_final_abl.log 2>&1
# ---------- 6) zero-shot Qwen3-Reranker-4B (int8) ----------
for ds in musique_dev hotpotqa_dev 2wiki_dev; do skip q3rr_$ds || { run $PY code/q3rr.py $ds > logs/q3rr_$ds.log 2>&1; mark q3rr_$ds; }; done
run $PY code/eval_select.py $D q3rr=results/scores/${D}_q3rr.pkl:score >> logs/eval_final_mq.log 2>&1
for ds in hotpotqa 2wiki; do
  run $PY code/eval_select.py ${ds}_dev lacg=ckpt/${ds}_lacr/pred_${ds}_dev.pkl:order jbce=ckpt/${ds}_jbce/pred_${ds}_dev.pkl:score \
    prove2=ckpt/${ds}_v2/pred_${ds}_dev.pkl:circuit beam=results/scores/${ds}_dev_beam.pkl:sets q3rr=results/scores/${ds}_dev_q3rr.pkl:score \
    rerank=results/scores/${ds}_dev_rerank.pkl:score bm25=results/scores/${ds}_dev_bm25.pkl:score > logs/eval_final_${ds}.log 2>&1
done
run $PY code/ans_eval.py lac=ckpt/mq_lacr/pred_${F}_ns1.pkl:ns lacg=ckpt/mq_lacr/pred_$F.pkl:order anscls=ckpt/mq_anscls/pred_$F.pkl:cls \
  prove2=ckpt/mq_v2/pred_$F.pkl:circuit jbce3=ckpt/mq_jbce3/pred_$F.pkl:score abl_mark=ckpt/abl_mark/pred_$F.pkl:order \
  abl_prefix=ckpt/abl_prefix/pred_$F.pkl:order abl_noroll=ckpt/abl_noroll/pred_$F.pkl:order abl_noworld=ckpt/abl_noworld/pred_$F.pkl:order \
  mq_lacr_s43=ckpt/mq_lacr_s43/pred_$F.pkl:order jbce3_s43=ckpt/mq_jbce3_s43/pred_$F.pkl:score > logs/ans_final.log 2>&1
# ---------- 7) LLM jobs (vLLM, int4 Qwen3-8B-AWQ) ----------
skip rank_mq || { run $PY code/llm_tools.py rank $D > logs/rank_mq.log 2>&1; mark rank_mq; }
run $PY code/eval_select.py $D rankgpt=results/scores/${D}_rankgpt_qwen3_8b.pkl:sets >> logs/eval_final_mq.log 2>&1
skip qa_mq || { run $PY code/llm_tools.py qa $D lac:k2,k4,t512,dec jbce3:k2,k4,t512,dec chain:k2,k4,t512 beam:k2,k4,t512,k5 q3rr:k2,k4,t512 \
  rankgpt:k2,k4,t512 setr:k2,k4,t512,k5 lacg:k2,k4,t512,dec point:k2,k4,t512,dec prove2:k2,k4,t512,dec oracle:k2 fullpool:k2 noctx:k2 > logs/qa_final_mq.log 2>&1; mark qa_mq; }
export QA_N=2000
for ds in hotpotqa 2wiki; do
  skip qa_$ds || { run $PY code/llm_tools.py qa ${ds}_dev lacg:k2,t384,dec jbce:k2,t384,dec beam:k2,t384,k5 q3rr:k2,t384 oracle:k2 > logs/qa_final_${ds}.log 2>&1; mark qa_$ds; }
done
unset QA_N
# ---------- 8) significance + tables ----------
run $PY code/signif.py $D lacg jbce3 > logs/signif_final.log 2>&1
for b in beam chain q3rr; do run $PY code/signif.py $D lacg $b >> logs/signif_final.log 2>&1; done
for ds in hotpotqa 2wiki; do for b in jbce beam; do run $PY code/signif.py ${ds}_dev lacg $b k2,k3,k4,t256,t384,t512 >> logs/signif_final.log 2>&1; done; done
run $PY code/gen_tables.py > logs/gen_tables.log 2>&1
touch done/ALL
echo "[$(date)] ALL DONE" >> $ST
