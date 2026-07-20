# nano-gpt-vs-bdh: reproducible harness

From-scratch implementations of a GPT-2-style **Transformer** and the **Dragon
Hatchling (BDH)** \[[arXiv:2509.26507](https://arxiv.org/abs/2509.26507)\],
trained identically on character-level TinyShakespeare and compared at
**matched parameter budgets**. This directory contains everything needed to
reproduce every number, table, and figure in `../paper.pdf`.

## What is here

| Path | What |
|---|---|
| `nanobdh/model_gpt.py` | the Transformer, heavily commented |
| `nanobdh/model_bdh.py` | BDH, adapted from the official reference impl |
| `nanobdh/train.py` | one shared training loop (`--model gpt\|bdh`) |
| `nanobdh/sample.py` | sample completions from a checkpoint |
| `nanobdh/analyze.py` | measures BDH activation sparsity |
| `nanobdh/finetune.py` | optional SFT to a chat assistant |
| `eval_base.py` | the eval harness (perplexity, text metrics, swap-controlled Claude judge) |
| `plot_eval.py`, `plot_training.py` | regenerate all figures from the raw JSONs |
| `gen_paper_assets.py` | emit `macros.tex` + `tables.tex` from the raw JSONs |
| `data/prepare.py` | download + tokenize TinyShakespeare |
| `docs/` | a 12-chapter from-scratch course (GPT and BDH), with SVG diagrams and a glossary |
| `reports/base_eval.json` | raw results of the **matched** run (headline) |
| `reports/base_eval_2x.json` | raw results of the **2x-size** run (the confound) |
| `out/gpt.pt`, `out/bdh.pt` | the exact trained checkpoints behind the numbers |
| `web/`, `serve.py` | a dual-model chat UI (optional, for the SFT assistants) |

## Setup

```bash
conda env create -f environment.yml   # creates env "nanobdh" (Python 3.11 + PyTorch)
conda activate nanobdh
python data/prepare.py                 # downloads TinyShakespeare -> data/*.pt
```

> macOS note: install torch, numpy, and matplotlib **all via pip** (mixing a
> conda-forge numpy with a pip torch triggers an OpenMP double-init crash). The
> provided `environment.yml` already does this.

## Reproduce the parameter-matched comparison

```bash
# 1. train both models from scratch, identical settings, only the arch differs
python -m nanobdh.train --model gpt                              # 818K params
python -m nanobdh.train --model bdh --neuron_dim_multiplier 16   # 803K params (matched)

# 2. run the full eval (perplexity + text metrics + 60 swap-controlled judge calls)
python eval_base.py                    # writes reports/base_eval.json
#   python eval_base.py --no_judge     # objective metrics only, no Claude calls

# 3. measure BDH activation sparsity
python -m nanobdh.analyze --what sparsity

# 4. regenerate figures + paper macros/tables
python plot_eval.py
python plot_training.py
python gen_paper_assets.py             # writes ../macros.tex and ../tables.tex
```

The `--neuron_dim_multiplier 16` flag is the whole parameter match: BDH's three
shared matrices scale as `multiplier x embed^2`, so halving the multiplier (32 ->
16) halves the big matrices and brings BDH from ~1.59M to ~803K parameters,
matching the Transformer's 818K. The 2x run that produced `base_eval_2x.json`
used the default `--neuron_dim_multiplier 32`.

## Reproducibility caveat

Training on Apple's MPS backend is **not bitwise deterministic** even with a
fixed seed, so a fresh training run will land near, but not exactly on, the
published numbers. The exact checkpoints behind the paper (`out/gpt.pt`,
`out/bdh.pt`) are therefore included so the eval is exactly reproducible without
retraining.
