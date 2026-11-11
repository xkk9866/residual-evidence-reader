# Residual Evidence Reader (RER)

Research code for complete evidence selection in multi-hop retrieval. RER reads a question, the current evidence set, and candidate passages together. It scores passages that are still needed and two terminal outcomes: the evidence is complete, or required support is outside the candidate pool.

## Repository contents

- `code/train3.py`: current residual reader, on-policy closure training, and inference.
- `code/prep_data.py`: conversion of MuSiQue, HotpotQA, and 2WikiMultihopQA to a common JSONL format.
- `code/decode3.py`, `code/ns_decode.py`: alternative decoding strategies.
- `code/eval_select.py`, `code/ans_eval.py`: evidence selection and answerability evaluation.
- Other `code/*.py` files: baselines, retrieval loop, analysis, and earlier model variants.
- `code/*.sh`: experiment launch scripts. These contain machine-specific paths and should be adapted before use.

Manuscript sources, figures, PDFs, datasets, model weights, predictions, and generated tables are deliberately excluded.

## Environment

Use Python 3.10+ with a PyTorch build appropriate for your CUDA version. Install the remaining packages with:

```bash
pip install -r requirements.txt
```

The main reader defaults to the Hugging Face encoder `BAAI/bge-reranker-v2-m3`. Some baseline and answer-generation scripts also require additional models or `vllm`.

## Data

Place source datasets under `data/raw/`. The preprocessing script expects the paths listed in `code/prep_data.py`. It writes normalized records to `data/proc/`. Before running it, change the local tokenizer path near the top of that script to a tokenizer available on your machine.

Each normalized JSONL record contains `id`, `question`, `answers`, `docs` (passages with `title` and `text`), `gold` (supporting passage indices), `gold_order`, and `ntok` (reader-token counts). The datasets and generated records are not included.

For example, after placing MuSiQue files at the paths expected by the script:

```bash
python code/prep_data.py musique
```

## Train and evaluate

Run commands from the repository root. A basic on-policy closure run is:

```bash
python code/train3.py --data musique --objective closure --out ckpt/musique_rer
```

The script reads `data/proc/musique_train.jsonl` and `data/proc/musique_dev.jsonl`, and saves the selected checkpoint as `ckpt/musique_rer/model.pt`. To run inference on a normalized dataset:

```bash
python code/train3.py --ckpt ckpt/musique_rer/model.pt --out ckpt/musique_rer --infer data/proc/musique_dev.jsonl
```

For the full set of options, run `python code/train3.py --help`. Checkpoint, prediction, and evaluation paths in the experiment scripts may need adjustment for your environment.

## Scope

This is research code. Training and evaluation need separately obtained datasets and pretrained models; the repository does not bundle them. No license is granted unless one is added by the authors.
