#!/bin/bash
# Local Beam Retrieval (official code, deberta-v3-base, official MuSiQue recipe; 5 epochs) on GPU5
cd /data/yaokang/proofset; PY=$PWD/.venv/bin/python
export HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=${GPU:-5} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
st() { echo "$(date) $*" >> logs/q40_status.log; }
E=/data/yaokang/models/dv3/deberta-v3-base
free() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i ${GPU:-5}; }
tr() { # ds epochs; waits for 7 GB free on the (shared) card and retries after an OOM
  until [ -f done/beam_$1 ]; do
  while [ "$(free)" -lt 7000 ]; do sleep 300; done
  (cd ext/beam_retriever && $PY train_beam_retriever.py --do_train --gradient_checkpointing --prefix beam_$1 --model_name $E --tokenizer_path $E \
    --dataset_type musique --train_file ../../data/beam_in/$1_train.jsonl --predict_file ../../data/beam_in/$1_sel400.jsonl \
    --train_batch_size 8 --learning_rate 1e-5 --mean_passage_len 120 --fp16 --beam_size 2 --predict_batch_size 1 \
    --num_train_epochs $2 --warmup-ratio 0.1 --log_period_ratio 0.01 --accumulate_gradients 8 --eval_period_ratio 0.5 \
    --output_dir /data/yaokang/proofset/ckpt/beam_$1) > logs/beam_$1.log 2>&1 && \
  $PY code/beam_pred.py $1 ckpt/beam_$1/checkpoint_best.pt $E >> logs/beam_$1.log 2>&1 && touch done/beam_$1 || { st retry $1; sleep 300; }
  done
  st beam $1
}
for d in $DS; do tr $d $EP; done
st ALL
