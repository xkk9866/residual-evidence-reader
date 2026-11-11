#!/bin/bash
# FINAL training queue (GPU 6 only).  One training at a time; q8_infer.sh runs in parallel and picks up ckpts via done/ markers.
# Final config: LACUNA = train3 (evidence-as-query, residual likelihood, lattice + 1-step learner states,
#               world deletion with distractor fill = pool size kept as in MuSiQue-Full), stage 2 from stage-1 joint reader.
cd /data/yaokang/proofset
export HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=6 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python
ST=logs/q7_status.log
mkdir -p done
run() { echo "[$(date)] START $*" >> $ST; "$@"; echo "[$(date)] END rc=$? $*" >> $ST; }
T3="$PY code/train3.py --data musique --init ckpt/mq_jbce/model.pt --epochs 1"
# 1) main model, same-backbone chain baseline, answerability classifier baseline (MuSiQue)
run $T3 --out ckpt/mq_lacr > logs/mq_lacr.log 2>&1; touch done/mq_lacr
run $PY code/chain_ce.py --data musique --init ckpt/musique_point/model.pt --out ckpt/mq_chain --epochs 1 > logs/mq_chain.log 2>&1; touch done/mq_chain
run $T3 --objective anscls --out ckpt/mq_anscls > logs/mq_anscls.log 2>&1; touch done/mq_anscls
# 2) HotpotQA / 2Wiki (20k training questions; stage 1 hotpotqa_jbce already trained)
run $PY code/train3.py --data hotpotqa --init ckpt/hotpotqa_jbce/model.pt --out ckpt/hotpotqa_lacr --epochs 1 --max_train 20000 > logs/hotpotqa_lacr.log 2>&1; touch done/hotpotqa_lacr
run $PY code/train2.py --data 2wiki --head bce --out ckpt/2wiki_jbce --epochs 1 --max_train 20000 > logs/2wiki_jbce.log 2>&1; touch done/2wiki_jbce
run $PY code/train3.py --data 2wiki --init ckpt/2wiki_jbce/model.pt --out ckpt/2wiki_lacr --epochs 1 --max_train 20000 > logs/2wiki_lacr.log 2>&1; touch done/2wiki_lacr
# 3) ablations (stage-2 variants from the same stage-1 reader, 1 epoch)
run $T3 --layout mark --out ckpt/abl_mark > logs/abl_mark.log 2>&1; touch done/abl_mark
run $T3 --states prefix --no_roll --out ckpt/abl_prefix > logs/abl_prefix.log 2>&1; touch done/abl_prefix
run $T3 --no_roll --out ckpt/abl_noroll > logs/abl_noroll.log 2>&1; touch done/abl_noroll
run $T3 --no_world --out ckpt/abl_noworld > logs/abl_noworld.log 2>&1; touch done/abl_noworld
# 4) second seed for LACUNA and for the compute-matched static control
run $T3 --seed 43 --out ckpt/mq_lacr_s43 > logs/mq_lacr_s43.log 2>&1; touch done/mq_lacr_s43
run $PY code/train2.py --data musique --head bce --ckpt ckpt/mq_jbce/model.pt --out ckpt/mq_jbce3_s43 --epochs 1 --seed 43 > logs/mq_jbce3_s43.log 2>&1; touch done/mq_jbce3_s43
touch done/TRAIN_ALL
echo "[$(date)] ALL DONE" >> $ST
