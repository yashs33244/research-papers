"""Evaluation harness for the two BASE models (GPT vs BDH), with Claude as judge.

Base models are text-COMPLETION models (they continue a prompt in Shakespeare
style), so we do NOT test "answer correctness". We compare completion quality with
several independent, cross-checking metrics:

  1. PERPLEXITY on held-out Shakespeare (data/val.pt): exp(mean cross-entropy).
     Lower = the model is less surprised by real text. We keep the per-window
     losses so we can show the whole distribution, not just the mean.

  2. AUTOMATIC TEXT METRICS on generated completions (pure counting, no model):
       - valid_word_frac : fraction of output words that are real English words
                           (235k-word system dictionary + a short archaic allowlist).
       - distinct2       : unique word-bigrams / total (diversity, anti-repetition).
       - repeat_frac     : fraction of words that repeat an earlier word.
       - speaker_labels  : count of "NAME:" speaker labels (Shakespeare structure).
       - avg_word_len    : mean characters per word.

  3. CLAUDE AS JUDGE, POSITION-BIAS CONTROLLED. For each prompt we ask the `claude`
     CLI to score both completions AND we ask it TWICE with the two completions
     SWAPPED (A<->B). A "decisive" win requires the SAME model to win in BOTH
     orderings; disagreement between orderings is recorded as "inconsistent"
     (position-dependent) and does not count as a win. This is stronger than a
     single randomized call. We then run a two-sided binomial test on the decisive
     wins to report statistical significance.

FAIRNESS NOTE (front and centre): BDH has ~2x the parameters of GPT here, so any
BDH win is confounded by size. A param-matched rerun is the rigorous follow-up.

Run:
    python eval_base.py                # full eval incl. Claude judge (60 claude calls)
    python eval_base.py --no_judge     # objective metrics only
    python eval_base.py --max_prompts 8  # quick smoke test
Writes reports/base_eval.json.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import subprocess
from math import comb

import torch

from nanobdh.tokenizer import CharTokenizer
from nanobdh.model_gpt import GPT, GPTConfig
from nanobdh.model_bdh import BDH, BDHConfig

_THIS = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(_THIS, "out")
DATA_DIR = os.path.join(_THIS, "data")
REPORTS = os.path.join(_THIS, "reports")
WORDS_PATH = "/usr/share/dict/words"

GEN = dict(max_new_tokens=140, temperature=0.8, top_k=40)

PROMPTS = [
    # famous real openings
    "To be, or not to be, that is the question:\n",
    "Now is the winter of our discontent\n",
    "Friends, Romans, countrymen, lend me your ears;\n",
    "If music be the food of love, play on;\n",
    "O Romeo, Romeo, wherefore art thou Romeo?\n",
    "All the world's a stage,\n",
    "Shall I compare thee to a summer's day?\n",
    "The quality of mercy is not strained;\n",
    "Cowards die many times before their deaths;\n",
    "Double, double toil and trouble;\n",
    # speaker cues (free generation in-format)
    "KING RICHARD III:\n",
    "ROMEO:\n",
    "HAMLET:\n",
    "First Citizen:\n",
    "MACBETH:\n",
    "JULIET:\n",
    "OTHELLO:\n",
    "KING LEAR:\n",
    "PROSPERO:\n",
    "IAGO:\n",
    # scene / dialogue cues
    "Enter the KING and QUEEN, with their train.\n",
    "My lord, I come to tell you that\n",
    "Enter GHOST.\n",
    "A room in the castle.\n",
    "Exeunt all but HAMLET.\n",
    # neutral English (stress test)
    "The weather today is\n",
    "Once upon a time there was a\n",
    "The meaning of life is\n",
    "In the beginning\n",
    "My name is\n",
]


def pick_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_base(name, device):
    ckpt = torch.load(os.path.join(OUT_DIR, f"{name}.pt"), map_location=device)
    tok = CharTokenizer(ckpt["vocab"])
    cfg = ckpt["config"]
    model = GPT(GPTConfig(**cfg)) if name == "gpt" else BDH(BDHConfig(**cfg))
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, tok, cfg, sum(p.numel() for p in model.parameters())


# ---------------------------------------------------------------- perplexity
@torch.no_grad()
def perplexity(model, device, block_size, max_chunks=400):
    val = torch.load(os.path.join(DATA_DIR, "val.pt"), map_location="cpu")
    n = min((len(val) - 1) // block_size, max_chunks)
    losses = []
    for i in range(n):
        s = i * block_size
        x = val[s:s + block_size].unsqueeze(0).to(device)
        y = val[s + 1:s + 1 + block_size].unsqueeze(0).to(device)
        _, loss = model(x, y)
        losses.append(loss.item())
    mean = sum(losses) / max(1, len(losses))
    return mean, float(torch.exp(torch.tensor(mean))), losses


# ---------------------------------------------------------------- auto metrics
def load_words():
    words = set()
    with open(WORDS_PATH, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            words.add(line.strip().lower())
    words.update(["thee", "thou", "thy", "thine", "hath", "doth", "art", "ere",
                  "oft", "wilt", "shalt", "tis", "twas", "o", "ay"])
    return words


SPEAKER_RE = re.compile(r"^[A-Z][A-Za-z ]{1,20}:", re.MULTILINE)


def text_metrics(text, words):
    toks = [re.sub(r"[^a-z]", "", w.lower()) for w in text.split()]
    toks = [t for t in toks if t]
    if not toks:
        return dict(valid_word_frac=0.0, distinct2=0.0, repeat_frac=0.0,
                    speaker_labels=0, avg_word_len=0.0, n_words=0)
    valid = sum(1 for t in toks if t in words)
    bigrams = list(zip(toks, toks[1:]))
    return dict(
        valid_word_frac=valid / len(toks),
        distinct2=(len(set(bigrams)) / len(bigrams)) if bigrams else 0.0,
        repeat_frac=1.0 - len(set(toks)) / len(toks),
        speaker_labels=len(SPEAKER_RE.findall(text)),
        avg_word_len=sum(len(t) for t in toks) / len(toks),
        n_words=len(toks),
    )


# ---------------------------------------------------------------- generation
@torch.no_grad()
def generate(model, tok, device, prompt, seed):
    torch.manual_seed(seed)
    ids = torch.tensor([tok.encode(prompt)], dtype=torch.long, device=device)
    out = model.generate(ids, GEN["max_new_tokens"], GEN["temperature"], GEN["top_k"])
    return tok.decode(out[0].tolist())[len(prompt):]


# ---------------------------------------------------------------- Claude judge
def judge_once(prompt, comp_a, comp_b):
    ask = (
        "You are judging two continuations produced by two tiny character-level "
        "language models trained only on Shakespeare. They are COMPLETION models, "
        "so judge writing quality, not factual answers.\n\n"
        f"PROMPT:\n{prompt}\n\nCOMPLETION A:\n{comp_a}\n\nCOMPLETION B:\n{comp_b}\n\n"
        "Score each 1-5 on: fluency (real, grammatical English words), coherence "
        "(reads as sensible connected text), shakespeare (verse, NAME: labels, "
        "archaic diction). Then pick the better completion.\n"
        "Reply with ONLY this JSON, no prose:\n"
        '{"A":{"fluency":N,"coherence":N,"shakespeare":N},'
        '"B":{"fluency":N,"coherence":N,"shakespeare":N},"winner":"A" or "B" or "tie"}'
    )
    try:
        res = subprocess.run(["claude", "-p", ask], capture_output=True, text=True,
                             timeout=150, stdin=subprocess.DEVNULL)
        m = re.search(r"\{.*\}", res.stdout.strip(), re.DOTALL)
        return json.loads(m.group(0)) if m else None
    except Exception as e:
        print("   judge error:", e)
        return None


def judge_pair(prompt, gc, bc):
    """Judge the pair TWICE with positions swapped. Decisive win = wins both orders."""
    v1 = judge_once(prompt, gc, bc)   # order 1: A=gpt, B=bdh
    v2 = judge_once(prompt, bc, gc)   # order 2: A=bdh, B=gpt
    if not v1 or not v2:
        return None
    w1 = {"A": "gpt", "B": "bdh", "tie": "tie"}.get(v1.get("winner"), "tie")
    w2 = {"A": "bdh", "B": "gpt", "tie": "tie"}.get(v2.get("winner"), "tie")
    if w1 == w2 and w1 in ("gpt", "bdh"):
        decisive, consistent = w1, True
    elif w1 == "tie" and w2 == "tie":
        decisive, consistent = "tie", True
    else:
        decisive, consistent = "inconsistent", False
    # average the two orderings' criterion scores per model
    def avg(a, b):
        return {k: (a[k] + b[k]) / 2 for k in a}
    gpt_scores = avg(v1["A"], v2["B"])
    bdh_scores = avg(v1["B"], v2["A"])
    return {"decisive": decisive, "consistent": consistent,
            "gpt": gpt_scores, "bdh": bdh_scores,
            "order1_winner": w1, "order2_winner": w2}


def binom_two_sided(k, n, p=0.5):
    if n == 0:
        return 1.0
    tail = sum(comb(n, i) for i in range(k, n + 1)) * (p ** n)
    return min(1.0, 2 * tail)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no_judge", action="store_true")
    ap.add_argument("--max_prompts", type=int, default=len(PROMPTS))
    args = ap.parse_args()

    device = pick_device()
    print(f"device: {device}")
    gpt, gtok, gcfg, gparams = load_base("gpt", device)
    bdh, btok, bcfg, bparams = load_base("bdh", device)
    print(f"gpt params {gparams:,}  |  bdh params {bparams:,}")
    words = load_words()

    g_loss, g_ppl, g_losses = perplexity(gpt, device, gcfg["block_size"])
    b_loss, b_ppl, b_losses = perplexity(bdh, device, bcfg["block_size"])
    print(f"perplexity  gpt {g_ppl:.2f}  |  bdh {b_ppl:.2f}  (over {len(g_losses)} windows)")

    prompts = PROMPTS[:args.max_prompts]
    rows = []
    for i, prompt in enumerate(prompts):
        gc = generate(gpt, gtok, device, prompt, seed=100 + i)
        bc = generate(bdh, btok, device, prompt, seed=100 + i)
        judge = None if args.no_judge else judge_pair(prompt, gc, bc)
        if not args.no_judge:
            print(f"  [{i+1:2d}/{len(prompts)}] {prompt.strip()[:30]:30s} "
                  f"-> {judge['decisive'] if judge else 'n/a'}")
        rows.append(dict(prompt=prompt,
                         gpt=dict(completion=gc, **text_metrics(gc, words)),
                         bdh=dict(completion=bc, **text_metrics(bc, words)),
                         judge=judge))

    def avg(key, model):
        return sum(r[model][key] for r in rows) / len(rows)

    judged = [r for r in rows if r["judge"]]
    decisive = {"gpt": 0, "bdh": 0, "tie": 0, "inconsistent": 0}
    for r in judged:
        decisive[r["judge"]["decisive"]] += 1
    n_dec = decisive["gpt"] + decisive["bdh"]
    winner_side = "bdh" if decisive["bdh"] >= decisive["gpt"] else "gpt"
    pval = binom_two_sided(decisive[winner_side], n_dec) if n_dec else 1.0

    def jscore(model, crit):
        vals = [r["judge"][model][crit] for r in judged]
        return sum(vals) / len(vals) if vals else 0.0

    report = {
        "params": {"gpt": gparams, "bdh": bparams},
        "perplexity": {"gpt": g_ppl, "bdh": b_ppl},
        "perplexity_windows": {"gpt": g_losses, "bdh": b_losses},
        "auto_metrics": {m: {k: avg(k, m) for k in
                             ("valid_word_frac", "distinct2", "repeat_frac",
                              "speaker_labels", "avg_word_len")}
                         for m in ("gpt", "bdh")},
        "judge": {
            "n_prompts": len(prompts),
            "n_judged": len(judged),
            "decisive": decisive,
            "consistency_rate": (sum(1 for r in judged if r["judge"]["consistent"]) /
                                 len(judged)) if judged else 0.0,
            "binomial_p": pval,
            "scores": {m: {c: jscore(m, c) for c in
                           ("fluency", "coherence", "shakespeare")}
                       for m in ("gpt", "bdh")},
        },
        "rows": rows,
    }
    os.makedirs(REPORTS, exist_ok=True)
    with open(os.path.join(REPORTS, "base_eval.json"), "w") as f:
        json.dump(report, f, indent=2)
    print("\nwrote reports/base_eval.json")
    print(f"decisive wins: {decisive}  | consistency {report['judge']['consistency_rate']:.0%}"
          f"  | binomial p = {pval:.2e}")


if __name__ == "__main__":
    main()
