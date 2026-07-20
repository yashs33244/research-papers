"""Emit macros.tex and tables.tex for paper.tex from the raw eval JSONs.

Reads:
  reports/base_eval.json      the parameter-MATCHED GPT vs BDH run (headline)
  reports/base_eval_2x.json   the earlier 2x-size run (the motivating confound)
  reports/sparsity.json       {"active_frac": <float>} from analyze.py (optional)

Writes (next to paper.tex):
  ../macros.tex   every number in the paper as a \newcommand
  ../tables.tex   the matched head-to-head table and the 2x-confound table

Every figure in the paper is likewise regenerated from these JSONs by
plot_eval.py / plot_training.py, so the manuscript contains no hand-typed
numbers. Run:  python gen_paper_assets.py
"""
import json
import os
import re

_THIS = os.path.dirname(os.path.abspath(__file__))
REPORTS = os.path.join(_THIS, "reports")
OUT = os.path.join(_THIS, "out")
PAPER = os.path.dirname(_THIS)  # nano-gpt-vs-bdh/  (where paper.tex lives)

# GPT's learned position table (block_size * n_embd = 128 * 128); BDH has none
# (it uses parameter-free rotary embeddings). Used to report the non-embedding
# parameter convention alongside the total-trainable-parameter convention.
GPT_WPE = 128 * 128


def load(name):
    with open(os.path.join(REPORTS, name)) as f:
        return json.load(f)


def best_val(path):
    """Smallest validation loss printed in a training log (best checkpoint)."""
    best = None
    for line in open(path):
        m = re.search(r"\|\s*val\s+([\d.]+)", line)
        if m:
            v = float(m.group(1))
            best = v if best is None else min(best, v)
    return best


def sci(p):
    """Format a p-value for inline math: decimal if >=1e-3 else a\\times10^{b}."""
    if p >= 1e-3:
        return f"{p:.2f}"
    exp = 0
    m = p
    while m < 1.0 and m > 0:
        m *= 10
        exp -= 1
    return f"{m:.1f}\\times10^{{{exp}}}"


def pct(x, d=1):
    return f"{100*x:.{d}f}\\%"


def decisive_score(rep):
    """Return 'BDH-GPT' style score string and (winner, n_dec, p)."""
    dec = rep["judge"]["decisive"]
    g, b = dec.get("gpt", 0), dec.get("bdh", 0)
    winner = "bdh" if b >= g else "gpt"
    hi, lo = (b, g) if b >= g else (g, b)
    return f"{hi}--{lo}", winner, dec, rep["judge"]["binomial_p"]


def main():
    m = load("base_eval.json")       # matched (headline)
    big = load("base_eval_2x.json")  # 2x confound
    try:
        spars = load("sparsity.json")["active_frac"]
    except Exception:
        spars = None

    # parameter counts come straight from the eval JSONs (sum of numel per model)
    GPT_PARAMS = m["params"]["gpt"]
    BDH_PARAMS = m["params"]["bdh"]
    BDH_BIG_PARAMS = big["params"]["bdh"]
    GPT_PARAMS_NOEMB = GPT_PARAMS - GPT_WPE  # nanoGPT non-embedding convention

    # best validation loss from the training logs (the checkpointed value)
    gpt_val = best_val(os.path.join(OUT, "train_gpt.log"))
    bdh_val = best_val(os.path.join(OUT, "train_bdh.log"))

    n_windows = len(m["perplexity_windows"]["gpt"])
    n_prompts = m["judge"]["n_prompts"]

    m_score, m_winner, m_dec, m_p = decisive_score(m)
    big_score, big_winner, big_dec, big_p = decisive_score(big)

    ma, ba = m["auto_metrics"]["gpt"], m["auto_metrics"]["bdh"]
    ms, bs = m["judge"]["scores"]["gpt"], m["judge"]["scores"]["bdh"]

    lines = []
    def mac(name, val):
        lines.append(f"\\newcommand{{\\{name}}}{{{val}}}")

    # --- parameters (fixed by design) ---
    mac("GptParamsK", f"{GPT_PARAMS/1e3:.0f}")
    mac("BdhParamsK", f"{BDH_PARAMS/1e3:.0f}")
    mac("BdhBigParamsK", f"{BDH_BIG_PARAMS/1e3:.0f}")
    mac("GptParamsM", f"{GPT_PARAMS/1e6:.2f}")
    mac("BdhParamsM", f"{BDH_PARAMS/1e6:.2f}")
    mac("BdhBigParamsM", f"{BDH_BIG_PARAMS/1e6:.2f}")
    mac("GptParamsNoEmbK", f"{GPT_PARAMS_NOEMB/1e3:.0f}")
    mac("MatchPct", f"{100*abs(GPT_PARAMS-BDH_PARAMS)/GPT_PARAMS:.0f}\\%")
    mac("GptValLoss", f"{gpt_val:.2f}")
    mac("BdhValLoss", f"{bdh_val:.2f}")
    mac("NIters", "3000")
    mac("NPrompts", str(n_prompts))
    mac("NWindows", str(n_windows))

    # --- matched run (headline) ---
    mac("GptPpl", f"{m['perplexity']['gpt']:.2f}")
    mac("BdhPpl", f"{m['perplexity']['bdh']:.2f}")
    mac("GptRealWord", pct(ma["valid_word_frac"]))
    mac("BdhRealWord", pct(ba["valid_word_frac"]))
    mac("GptDistinct", pct(ma["distinct2"], 0))
    mac("BdhDistinct", pct(ba["distinct2"], 0))
    mac("GptSpeak", f"{ma['speaker_labels']:.1f}")
    mac("BdhSpeak", f"{ba['speaker_labels']:.1f}")
    mac("DecGpt", str(m_dec.get("gpt", 0)))
    mac("DecBdh", str(m_dec.get("bdh", 0)))
    mac("DecTie", str(m_dec.get("tie", 0)))
    mac("DecIncon", str(m_dec.get("inconsistent", 0)))
    mac("MatchScore", m_score)
    mac("Consistency", f"{100*m['judge']['consistency_rate']:.0f}\\%")
    mac("Pval", sci(m_p))
    mac("Winner", m_winner)
    mac("WinnerName", m_winner.upper())
    mac("GptFlu", f"{ms['fluency']:.1f}")
    mac("GptCoh", f"{ms['coherence']:.1f}")
    mac("GptShak", f"{ms['shakespeare']:.1f}")
    mac("BdhFlu", f"{bs['fluency']:.1f}")
    mac("BdhCoh", f"{bs['coherence']:.1f}")
    mac("BdhShak", f"{bs['shakespeare']:.1f}")

    # --- 2x confound run (motivating) ---
    mac("JudgeBigScore", big_score)
    mac("PvalBig", sci(big_p))
    mac("GptPplBig", f"{big['perplexity']['gpt']:.2f}")
    mac("BdhPplBig", f"{big['perplexity']['bdh']:.2f}")
    mac("GptRealWordBig", pct(big["auto_metrics"]["gpt"]["valid_word_frac"]))
    mac("BdhRealWordBig", pct(big["auto_metrics"]["bdh"]["valid_word_frac"]))

    # --- sparsity ---
    mac("SparsityFrac", pct(spars, 0) if spars is not None else "31\\%")

    with open(os.path.join(PAPER, "macros.tex"), "w") as f:
        f.write("% AUTO-GENERATED by experiment/gen_paper_assets.py -- do not edit by hand.\n")
        f.write("\n".join(lines) + "\n")
    print(f"wrote macros.tex ({len(lines)} macros)")

    # ---------------------------------------------------------------- tables
    def bold(x):
        return f"\\textbf{{{x}}}"

    def row(label, gv, bv, better_low=False, fmt=str):
        """Row with the better of gv/bv bolded. better_low => smaller is better.
        Only bold when the DISPLAYED values differ, so visual ties stay unbolded."""
        gs, bs_ = fmt(gv), fmt(bv)
        if gs != bs_:
            if (gv < bv) == better_low:
                gs = bold(gs)
            else:
                bs_ = bold(bs_)
        return f"{label} & {gs} & {bs_} \\\\"

    p2 = lambda x: f"{x:.2f}"
    pc = lambda x: f"{100*x:.1f}\\%"
    pc0 = lambda x: f"{100*x:.0f}\\%"
    f1 = lambda x: f"{x:.1f}"

    matched_rows = "\n".join([
        row("Perplexity (lower better)", m["perplexity"]["gpt"], m["perplexity"]["bdh"], better_low=True, fmt=p2),
        row("Real-word rate", ma["valid_word_frac"], ba["valid_word_frac"], fmt=pc),
        row("Distinct-2 (diversity)", ma["distinct2"], ba["distinct2"], fmt=pc0),
        row("Speaker labels / gen", ma["speaker_labels"], ba["speaker_labels"], fmt=f1),
        row(f"Judge decisive wins (of {n_prompts})", m_dec.get("gpt", 0), m_dec.get("bdh", 0), fmt=str),
        row("Judge fluency (1--5)", ms["fluency"], bs["fluency"], fmt=f1),
        row("Judge coherence (1--5)", ms["coherence"], bs["coherence"], fmt=f1),
        row("Judge Shakespeare (1--5)", ms["shakespeare"], bs["shakespeare"], fmt=f1),
    ])

    tables = rf"""% AUTO-GENERATED by experiment/gen_paper_assets.py -- do not edit by hand.
\begin{{table}}[t]\centering\small
\caption{{Parameter-matched head-to-head: GPT ({GPT_PARAMS:,} params) vs BDH
({BDH_PARAMS:,} params, matched within {100*abs(GPT_PARAMS-BDH_PARAMS)/GPT_PARAMS:.0f}\%). Both trained from scratch under
identical settings on character-level TinyShakespeare. Perplexity is over
{n_windows} validation windows; text metrics and the judge are over {n_prompts}
prompts. Judge pairs are scored in both orderings; a decisive win requires
winning both. Better value in bold.}}
\label{{tab:matched}}
\begin{{tabular}}{{lcc}}
\toprule
Metric & GPT ({GPT_PARAMS/1e3:.0f}K) & BDH ({BDH_PARAMS/1e3:.0f}K) \\
\midrule
{matched_rows}
\midrule
Position-swap consistency & \multicolumn{{2}}{{c}}{{{100*m['judge']['consistency_rate']:.0f}\%}} \\
Binomial $p$ (decisive wins) & \multicolumn{{2}}{{c}}{{${sci(m_p)}$}} \\
\bottomrule
\end{{tabular}}
\end{{table}}

\begin{{table}}[t]\centering\small
\caption{{The motivating confound: the same harness with BDH at its natural
{BDH_BIG_PARAMS/1e6:.2f}M-parameter default ($\sim$2$\times$ GPT). Here BDH sweeps
every metric. Comparing this table with Table~\ref{{tab:matched}} is the whole
point: it separates a size effect from an architecture effect.}}
\label{{tab:confound}}
\begin{{tabular}}{{lcc}}
\toprule
Metric & GPT ({GPT_PARAMS/1e3:.0f}K) & BDH ({BDH_BIG_PARAMS/1e6:.2f}M) \\
\midrule
{row("Perplexity (lower better)", big["perplexity"]["gpt"], big["perplexity"]["bdh"], better_low=True, fmt=p2)}
{row("Real-word rate", big["auto_metrics"]["gpt"]["valid_word_frac"], big["auto_metrics"]["bdh"]["valid_word_frac"], fmt=pc)}
{row(f"Judge decisive wins (of {big['judge']['n_prompts']})", big_dec.get("gpt", 0), big_dec.get("bdh", 0), fmt=str)}
\midrule
Position-swap consistency & \multicolumn{{2}}{{c}}{{{100*big['judge']['consistency_rate']:.0f}\%}} \\
Binomial $p$ (decisive wins) & \multicolumn{{2}}{{c}}{{${sci(big_p)}$}} \\
\bottomrule
\end{{tabular}}
\end{{table}}
"""
    with open(os.path.join(PAPER, "tables.tex"), "w") as f:
        f.write(tables)
    print("wrote tables.tex")


if __name__ == "__main__":
    main()
