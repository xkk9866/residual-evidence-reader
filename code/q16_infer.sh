#!/bin/bash
# GPU6 INFERENCE lane (runs beside the training lane; small memory). waits for done/tr_<name>.
cd /data/yaokang/proofset
export HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=6 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python
ST=logs/q16_status.log
run() { echo "[$(date)] START $*" >> $ST; "$@"; echo "[$(date)] END rc=$? $*" >> $ST; }
waitfor() { while [ ! -f done/tr_$1 ]; do sleep 60; done; }
mkdir -p done
MQ=data/proc/musique_dev.jsonl,data/proc/musique_full_dev.jsonl
OP=data/proc/openpool_musique_N20.jsonl,data/proc/openpool_hotpotqa_N20.jsonl,data/proc/openpool_2wikimultihopqa_N20.jsonl
waitfor mq_lacp; while [ ! -f done/inf_mq_lacp ]; do sleep 60; done
# open corpus: single-shot pools for all same-backbone readers, closed loop for LACUNA-PP
[ -f done/inf_open_lacp ] || { run $PY code/train3.py --objective point --ckpt ckpt/mq_lacp/model.pt --out ckpt/mq_lacp --infer $OP > logs/inf_open_lacp.log 2>&1; touch done/inf_open_lacp; }
[ -f done/inf_open_jbce3 ] || { run $PY code/train2.py --head bce --ckpt ckpt/mq_jbce3/model.pt --out ckpt/mq_jbce3 --infer $OP > logs/inf_open_jbce3.log 2>&1; touch done/inf_open_jbce3; }
[ -f done/inf_open_lacr ] || { run $PY code/train3.py --ckpt ckpt/mq_lacr/model.pt --out ckpt/mq_lacr --infer $OP > logs/inf_open_lacr.log 2>&1; touch done/inf_open_lacr; }
[ -f done/inf_open_chain ] || { run $PY code/chain_ce.py --ckpt ckpt/mq_chain/model.pt --out ckpt/mq_chain --infer $OP > logs/inf_open_chain.log 2>&1; touch done/inf_open_chain; }
for ds in musique hotpotqa 2wikimultihopqa; do
  for m in adaptive fixed; do
    [ -f done/loop_${ds}_$m ] || { run $PY code/train3.py --objective point --ckpt ckpt/mq_lacp/model.pt --out ckpt/mq_lacp --openloop $ds --loop_mode $m --rounds 2 > logs/loop_${ds}_$m.log 2>&1; touch done/loop_${ds}_$m; }
  done
done
waitfor mq_jbce4; [ -f done/inf_mq_jbce4 ] || { run $PY code/train2.py --head bce --ckpt ckpt/mq_jbce4/model.pt --out ckpt/mq_jbce4 --infer $MQ > logs/inf_mq_jbce4.log 2>&1; touch done/inf_mq_jbce4; }
for ds in hotpotqa 2wiki; do
  a=${ds}_lacp; waitfor $a
  [ -f done/inf_$a ] || { run $PY code/train3.py --objective point --ckpt ckpt/$a/model.pt --out ckpt/$a --infer data/proc/${ds}_dev.jsonl > logs/inf_$a.log 2>&1; touch done/inf_$a; }
done
for a in mq_lacp_s43 mq_lacp_s44 pabl_ce1 pabl_mark pabl_noworld pabl_noroll pabl_prefix; do
  waitfor $a
  L=""; [ $a = pabl_mark ] && L="--layout mark"
  [ -f done/inf_$a ] || { run $PY code/train3.py --objective point $L --ckpt ckpt/$a/model.pt --out ckpt/$a --infer $MQ > logs/inf_$a.log 2>&1; touch done/inf_$a; }
done
touch done/INFER16_ALL
echo "[$(date)] ALL DONE" >> $ST
