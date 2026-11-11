#!/bin/bash
# GPU6 final queue (single, serial except small inference beside training). resumable via done/
cd /data/yaokang/proofset
export HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=6 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python
ST=logs/q17_status.log
run() { echo "[$(date)] START $*" >> $ST; "$@"; echo "[$(date)] END rc=$? $*" >> $ST; }
mkdir -p done
OP=data/proc/openpool_musique_N20.jsonl,data/proc/openpool_hotpotqa_N20.jsonl,data/proc/openpool_2wikimultihopqa_N20.jsonl
MQ=data/proc/musique_dev.jsonl,data/proc/musique_full_dev.jsonl
# (a) inference beside the running jbce4 training: open-corpus pools + closed loop for LACUNA
[ -f done/inf_open_lacr ] || { run $PY code/train3.py --ckpt ckpt/mq_lacr/model.pt --out ckpt/mq_lacr --infer $OP > logs/inf_open_lacr.log 2>&1; touch done/inf_open_lacr; }
[ -f done/inf_open_jbce3 ] || { run $PY code/train2.py --head bce --ckpt ckpt/mq_jbce3/model.pt --out ckpt/mq_jbce3 --infer $OP > logs/inf_open_jbce3.log 2>&1; touch done/inf_open_jbce3; }
[ -f done/inf_open_chain ] || { run $PY code/chain_ce.py --ckpt ckpt/mq_chain/model.pt --out ckpt/mq_chain --infer $OP > logs/inf_open_chain.log 2>&1; touch done/inf_open_chain; }
for ds in musique hotpotqa 2wikimultihopqa; do for m in adaptive fixed; do
  [ -f done/loopr_${ds}_$m ] || { run $PY code/train3.py --ckpt ckpt/mq_lacr/model.pt --out ckpt/mq_lacr --openloop $ds --loop_mode $m --rounds 2 > logs/loopr_${ds}_$m.log 2>&1; touch done/loopr_${ds}_$m; }
done; done
# (b) training: wait for jbce4, then the clean loss-only ablation (same states + terminals, single-positive CE)
while pgrep -f "^.venv/bin/python code/train2.py --data musique --head bce --ckpt ckpt/mq_jbce/model.pt --out ckpt/mq_jbce4" >/dev/null; do sleep 30; done
touch done/tr_mq_jbce4
[ -f done/inf_mq_jbce4 ] || { run $PY code/train2.py --head bce --ckpt ckpt/mq_jbce4/model.pt --out ckpt/mq_jbce4 --infer $MQ > logs/inf_mq_jbce4.log 2>&1; touch done/inf_mq_jbce4; }
a=abl_ce1
[ -f done/tr_$a ] || { run $PY code/train3.py --data musique --objective ce1 --init ckpt/mq_jbce/model.pt --epochs 1 --out ckpt/$a > logs/$a.log 2>&1; touch done/tr_$a; }
[ -f done/inf_$a ] || { run $PY code/train3.py --ckpt ckpt/$a/model.pt --out ckpt/$a --infer $MQ > logs/inf_$a.log 2>&1; touch done/inf_$a; }
# (c) re-infer LACUNA on dev with halting/verdict fields + timing (true early-stop cost measured separately in eval)
[ -f done/inf_lacr_v ] || { run $PY code/train3.py --ckpt ckpt/mq_lacr/model.pt --out ckpt/mq_lacr_v --infer $MQ > logs/inf_lacr_v.log 2>&1; touch done/inf_lacr_v; }
touch done/Q17_GPU
echo "[$(date)] GPU PART DONE" >> $ST
