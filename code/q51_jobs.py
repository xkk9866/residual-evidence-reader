"""Round q51: the remaining GPU work of the final paper as one queue for code/sched2.py on ALL cards (SCHED_GPUS=0..7).
  python code/q51_jobs.py && SCHED_GPUS=0,1,2,3,4,5,6,7 nohup .venv/bin/python code/sched2.py sched/q51_jobs.json > logs/sched2.out 2>&1 &
Every job is idempotent (finished outputs are skipped by the tools themselves or by the sched/*.ok markers)."""
import json, os

PY = '.venv/bin/python'
QA = f'{PY} code/llm_tools.py qamulti'
MQ = 'data/proc/musique_dev.jsonl,data/proc/musique_full_dev.jsonl'
E = f'{PY} code/eval_select.py'
PFX = 'q51_'
J = []


def job(name, cmd, need, deps=(), prio=50, llm=False, soft=False):
    J.append({'name': PFX + name, 'cmd': cmd, 'need': need, 'deps': [PFX + d for d in deps], 'prio': prio, 'llm': llm, 'soft': soft})


# ---- IRCoT (last corpus) and the QA of the retrained Beam sets + the analysis stop sets (int4 Qwen3-8B), then Qwen3-4B
# (no own redirect: the scheduler's log of the job is what its OOM detection reads)
job('ircot_2wiki', f'[ -f results/metrics/ircot_2wikimultihopqa.json ] || MAXLEN=8192 {PY} code/ircot.py 2wikimultihopqa', 12000, prio=1, llm=True)
job('qa_beam', f'RTAG=qwen3_8b {QA} "musique_dev|beam:k2,k4,t512,k5" "musique_dev|beamk:k2,k4,t512,k5" "musique_dev|beam_dec:dec" "musique_dev|delta_dec:dec" "musique_dev|lacc_loo_dec:dec" > logs/qa51_beam.log 2>&1 && '
    f'QA_N=2000 RTAG=qwen3_8b {QA} "hotpotqa_dev|beam:k2,t384,k5" "hotpotqa_dev|beamk:k2,t384" "2wiki_dev|beam:k2,t384,k5" "2wiki_dev|beamk:k2,t384" >> logs/qa51_beam.log 2>&1', 10500, prio=1, llm=True)
job('qa4_beam', f'RTAG=qwen3_4b READER=$HOME/models/Qwen3-4B-AWQ {QA} "musique_dev|beam:k4" "musique_dev|beamk:k4" > logs/qa51_beam4.log 2>&1', 8000, prio=2, llm=True)

# ---- two-sided residual reader (analysis): third stage-2 seed, and a 2-epoch run with the negated-relevance release initialisation
TR = f'{PY} code/train3.py --data musique --init ckpt/mq_jbce/model.pt --objective delta'
job('tr_delta_s44', f'[ -f ckpt/mq_delta_s44/pred_musique_full_dev.pkl ] || ({TR} --epochs 1 --seed 44 --out ckpt/mq_delta_s44 > logs/mq_delta_s44.log 2>&1 && '
    f'{PY} code/train3.py --objective delta --ckpt ckpt/mq_delta_s44/model.pt --out ckpt/mq_delta_s44 --infer {MQ} >> logs/mq_delta_s44.log 2>&1)', 13500, prio=4)
job('tr_delta_neg2', f'[ -f ckpt/mq_delta_neg2/pred_musique_full_dev.pkl ] || ({TR} --epochs 2 --rem_init neg --save_all --out ckpt/mq_delta_neg2 > logs/mq_delta_neg2.log 2>&1 && '
    f'{PY} code/train3.py --objective delta --ckpt ckpt/mq_delta_neg2/model.pt --out ckpt/mq_delta_neg2 --infer {MQ} >> logs/mq_delta_neg2.log 2>&1)', 13500, prio=5)

# ---- proof identification (round 12): the exact state of the reader's own walk must carry the largest terminal mass.
#      Two-sided reader with identification (main candidate), one-sided reader with identification (ablation: identification
#      without release); both one stage-2 epoch from the stage-1 reader, decoded with the identification rule.
INF = lambda ck, out, extra: f'{PY} code/train3.py {extra} --ckpt {ck} --out {out} --infer {MQ}'
job('tr_delta_id', f'[ -f ckpt/mq_delta_id/pred_musique_full_dev.pkl ] || ({TR} --epochs 1 --ident 1 --out ckpt/mq_delta_id > logs/mq_delta_id.log 2>&1 && '
    f'{INF("ckpt/mq_delta_id/model.pt", "ckpt/mq_delta_id", "--objective delta --stop ident")} >> logs/mq_delta_id.log 2>&1)', 13500, prio=0)
job('tr_lacc_id', f'[ -f ckpt/mq_lacc_id/pred_musique_full_dev.pkl ] || ({PY} code/train3.py --data musique --init ckpt/mq_jbce/model.pt --epochs 1 --objective closure --ident 1 --roll walk --walk 5 --out ckpt/mq_lacc_id > logs/mq_lacc_id.log 2>&1 && '
    f'{INF("ckpt/mq_lacc_id/model.pt", "ckpt/mq_lacc_id", "--objective closure --stop ident --roll walk --walk 5")} >> logs/mq_lacc_id.log 2>&1)', 13500, prio=0)
# ---- training budget: the main reader continued for two more stage-2 epochs (3 in total), and the static control for two more (+4)
job('tr_lacc_e3', f'[ -f ckpt/mq_lacc_e3/pred_musique_full_dev.pkl ] || ({PY} code/train3.py --data musique --ckpt ckpt/mq_lacc/model.pt --epochs 2 --objective closure --out ckpt/mq_lacc_e3 > logs/mq_lacc_e3.log 2>&1 && '
    f'{INF("ckpt/mq_lacc_e3/model.pt", "ckpt/mq_lacc_e3", "--objective closure")} >> logs/mq_lacc_e3.log 2>&1)', 13500, prio=1)
job('tr_jbce5', f'[ -f ckpt/mq_jbce5/pred_musique_full_dev.pkl ] || ({PY} code/train2.py --data musique --head bce --ckpt ckpt/mq_jbce4/model.pt --epochs 2 --out ckpt/mq_jbce5 > logs/mq_jbce5.log 2>&1 && '
    f'{PY} code/train2.py --head bce --ckpt ckpt/mq_jbce5/model.pt --out ckpt/mq_jbce5 --infer {MQ} >> logs/mq_jbce5.log 2>&1)', 13500, prio=6)
# the identification rule applied to the existing readers (decoding only; separates the rule from the training term)
job('inf_delta2_idrule', f'[ -f ckpt/mq_delta2_idrule/pred_musique_full_dev.pkl ] || {INF("ckpt/mq_delta2/model.pt", "ckpt/mq_delta2_idrule", "--objective delta --stop ident")}', 5500, prio=2)
job('inf_lacc_idrule', f'[ -f ckpt/mq_lacc_idrule/pred_musique_full_dev.pkl ] || {INF("ckpt/mq_lacc/model.pt", "ckpt/mq_lacc_idrule", "--objective closure --stop ident")}', 5500, prio=2)

# ---- Beam Retrieval on MuSiQue: continue the official training from the 3-epoch checkpoint for 2 more epochs (5 in total)
B3 = 'ckpt/beam_musique/11-03-2026/beam_musique-seed42-bsz8-fp16True-lr1e-05-decay0.0-warm0.1-valbsz1/checkpoint_best.pt'
DV3 = '/data/yaokang/models/dv3/deberta-v3-base'
PYA = '/data/yaokang/proofset/.venv/bin/python'      # absolute: the official trainer runs from ext/beam_retriever
job('beam_cont', f'[ -f results/scores/musique_beam5L.pkl ] || ((cd ext/beam_retriever && {PYA} train_beam_retriever.py --do_train --gradient_checkpointing --prefix beam_musique5 --model_name {DV3} --tokenizer_path {DV3} '
    f'--init_checkpoint ../../{B3} --dataset_type musique --train_file ../../data/beam_in/musique_train.jsonl --predict_file ../../data/beam_in/musique_sel400.jsonl '
    f'--train_batch_size 8 --learning_rate 1e-5 --mean_passage_len 120 --fp16 --beam_size 2 --predict_batch_size 1 --num_train_epochs 2 --warmup-ratio 0.1 --log_period_ratio 0.01 '
    f'--accumulate_gradients 8 --eval_period_ratio 0.5 --output_dir /data/yaokang/proofset/ckpt/beam_musique5) > logs/beam_musique5.log 2>&1 && '
    f'{PY} code/beam_pred.py musique $(ls -t ckpt/beam_musique5/*/*/checkpoint_best.pt | head -n 1) {DV3} beam5 >> logs/beam_musique5.log 2>&1 && '
    f'{E} musique_dev beam5=results/scores/musique_beam5L.pkl:sets beam5k=results/scores/musique_beam5K.pkl:sets >> logs/beam_musique5.log 2>&1)', 7000, prio=3)

# ---- QA (int4 Qwen3-8B) on the self-terminated sets of the round-12 readers: the sets are written by delta_stats.py in final51.sh,
#      so the evaluation pass runs first; every spec is included only if its set file exists
QA_ID = ' '.join(f'$([ -f results/sel/musique_dev_{n}_dec.json ] && echo "musique_dev|{n}_dec:dec")' for n in ['lacc_e3', 'lacc_idb', 'delta_neg2', 'delta_id', 'delta_s44'])
job('qa_id', f'bash code/final51.sh > logs/final51.log 2>&1; MAXLEN=3072 RTAG=qwen3_8b {QA} {QA_ID} > logs/qa51_id.log 2>&1', 8500,
    ['tr_delta_id', 'tr_lacc_id', 'tr_lacc_e3', 'tr_delta_neg2', 'tr_delta_s44'], prio=3, llm=True, soft=True)

# ---- final CPU pass (every evaluation again, then the tables); soft: runs whatever has finished
J.append({'name': PFX + 'final', 'cmd': 'bash code/final51.sh > logs/final51.log 2>&1', 'need': 0, 'deps': [j['name'] for j in J], 'prio': 99, 'llm': False, 'soft': True})
os.makedirs('sched', exist_ok=True)
json.dump(J, open('sched/q51_jobs.json', 'w'), indent=1)
print(len(J), 'jobs')
