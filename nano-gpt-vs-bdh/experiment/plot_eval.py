"""Turn the raw eval JSONs into high-quality, data-driven comparison figures.

Reads reports/base_eval.json (the parameter-MATCHED run) and, where present,
reports/base_eval_2x.json (the earlier 2x-size run) and writes:

  eval_objective.png        perplexity distribution + automatic text metrics
  eval_judge.png            decisive judge wins (swap-controlled) + criterion scores
  eval_scorecard.png        one-glance summary; 4th tile shows the MATCHED budgets
  eval_matched_vs_confound.png   the money figure: 2x result vs matched result

All titles are computed from the data, so the figures state whatever actually
happened (BDH win, GPT win, or a wash) rather than a hard-coded story.
"""
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

_THIS = os.path.dirname(os.path.abspath(__file__))
REPORTS = os.path.join(_THIS, "reports")

INK="#1A1D23"; SUB="#5A6472"; GRID="#E6E9EE"; PANEL="#F7F8FA"
GPTC="#2E86AB"; BDHC="#E9A12E"; WIN="#2A9D8F"; RED="#D1495B"

plt.rcParams.update({
    "font.family": "Helvetica", "axes.edgecolor": SUB, "axes.linewidth": 0.9,
    "text.color": INK, "xtick.color": INK, "ytick.color": INK,
    "savefig.dpi": 170, "figure.dpi": 110, "savefig.bbox": "tight",
})

R = json.load(open(os.path.join(REPORTS, "base_eval.json")))
_big_path = os.path.join(REPORTS, "base_eval_2x.json")
BIG = json.load(open(_big_path)) if os.path.exists(_big_path) else None

GP, BP = R["params"]["gpt"], R["params"]["bdh"]
LGPT = f"GPT ({GP/1e3:.0f}K)"
LBDH = f"BDH ({BP/1e3:.0f}K)"

def dec_score(rep):
    d = rep["judge"]["decisive"]; g, b = d.get("gpt",0), d.get("bdh",0)
    win = "BDH" if b > g else ("GPT" if g > b else "tie")
    return g, b, win, rep["judge"]["binomial_p"], rep["judge"]["consistency_rate"]

def style(ax):
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.set_axisbelow(True); ax.grid(axis="y", color=GRID, lw=1); ax.tick_params(length=0)

def suptitle(fig, t, s):
    fig.text(0.5, 0.985, t, ha="center", va="top", fontsize=16, fontweight="bold", color=INK)
    fig.text(0.5, 0.928, s, ha="center", va="top", fontsize=10, color=SUB)

def legend(fig, y=0.875):
    h = [plt.Rectangle((0,0),1,1,color=GPTC), plt.Rectangle((0,0),1,1,color=BDHC)]
    fig.legend(h, [LGPT, LBDH], loc="upper center", bbox_to_anchor=(0.5, y),
               ncol=2, frameon=False, fontsize=10.5, columnspacing=2.0, handlelength=1.3)

# ============================================================ objective
def fig_objective():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.6, 5.9))
    style(ax1)
    gl = R["perplexity_windows"]["gpt"]; bl = R["perplexity_windows"]["bdh"]
    bp = ax1.boxplot([gl, bl], positions=[0,1], widths=0.5, patch_artist=True,
                     medianprops=dict(color=INK, lw=1.6), showfliers=False)
    for patch, c in zip(bp["boxes"], [GPTC, BDHC]): patch.set_facecolor(c); patch.set_alpha(0.85)
    ax1.set_xticks([0,1]); ax1.set_xticklabels([LGPT, LBDH])
    ax1.set_ylabel("per-window cross-entropy on val\n(lower is better)")
    ax1.set_title(f"Perplexity {R['perplexity']['gpt']:.2f} vs {R['perplexity']['bdh']:.2f}"
                  f"  ({len(gl)} windows)", fontsize=12, fontweight="bold", pad=10)

    style(ax2)
    am = R["auto_metrics"]
    labels = ["real words", "distinct-2", "speaker\nlabels/gen"]
    g = [am["gpt"]["valid_word_frac"]*100, am["gpt"]["distinct2"]*100, am["gpt"]["speaker_labels"]*10]
    d = [am["bdh"]["valid_word_frac"]*100, am["bdh"]["distinct2"]*100, am["bdh"]["speaker_labels"]*10]
    raw_g = [am["gpt"]["valid_word_frac"]*100, am["gpt"]["distinct2"]*100, am["gpt"]["speaker_labels"]]
    raw_d = [am["bdh"]["valid_word_frac"]*100, am["bdh"]["distinct2"]*100, am["bdh"]["speaker_labels"]]
    x = np.arange(3); w = 0.36
    b1 = ax2.bar(x-w/2, g, w, color=GPTC); b2 = ax2.bar(x+w/2, d, w, color=BDHC)
    ax2.set_xticks(x); ax2.set_xticklabels(labels)
    ax2.set_ylim(0, 112); ax2.set_ylabel("percent (real words, distinct-2) / count")
    fmts = ["{:.0f}%", "{:.0f}%", "{:.1f}"]
    for grp, raw in ((b1, raw_g), (b2, raw_d)):
        for bb, rv, fm in zip(grp, raw, fmts):
            ax2.annotate(fm.format(rv), (bb.get_x()+bb.get_width()/2, bb.get_height()+1.2),
                         ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax2.set_title("Generation quality (automatic, higher better)", fontsize=12, fontweight="bold", pad=10)

    suptitle(fig, f"Objective metrics at matched size ({GP/1e3:.0f}K vs {BP/1e3:.0f}K params)",
             "Validation perplexity spread, plus automatic word-level quality of the generated completions.")
    legend(fig)
    fig.subplots_adjust(top=0.76, bottom=0.12, wspace=0.24, left=0.08, right=0.97)
    fig.savefig(os.path.join(REPORTS, "eval_objective.png")); plt.close(fig)
    print("wrote eval_objective.png")

# ============================================================ judge
def fig_judge():
    j = R["judge"]; dec = j["decisive"]; sc = j["scores"]
    g, b, win, p, cons = dec_score(R)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.6, 5.9))
    style(ax1)
    cats = [("gpt", LGPT, GPTC), ("bdh", LBDH, BDHC), ("tie", "tie", SUB), ("inconsistent", "incon-\nsistent", RED)]
    vals = [dec.get(k, 0) for k,_,_ in cats]
    bars = ax1.bar(range(4), vals, 0.6, color=[c for _,_,c in cats])
    ax1.set_xticks(range(4)); ax1.set_xticklabels([l for _,l,_ in cats])
    ax1.set_ylim(0, max(vals)*1.2 if max(vals) else 1); ax1.set_ylabel("prompts won (both orderings)")
    for bb in bars:
        ax1.annotate(f"{bb.get_height():.0f}", (bb.get_x()+bb.get_width()/2, bb.get_height()+0.2),
                     ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax1.set_title(f"Decisive wins  (n={j['n_judged']}, judged both orderings)",
                  fontsize=12, fontweight="bold", pad=10)

    style(ax2)
    crits = ["fluency", "coherence", "shakespeare"]
    x = np.arange(3); w = 0.36
    gg = [sc["gpt"][c] for c in crits]; dd = [sc["bdh"][c] for c in crits]
    b1 = ax2.bar(x-w/2, gg, w, color=GPTC); b2 = ax2.bar(x+w/2, dd, w, color=BDHC)
    ax2.set_xticks(x); ax2.set_xticklabels([c.capitalize() for c in crits])
    ax2.set_ylim(0, 5.4); ax2.set_ylabel("mean judge score (1-5)")
    for bb in list(b1)+list(b2):
        ax2.annotate(f"{bb.get_height():.1f}", (bb.get_x()+bb.get_width()/2, bb.get_height()+0.06),
                     ha="center", va="bottom", fontsize=10.5, fontweight="bold")
    ax2.set_title("Mean scores by criterion", fontsize=12, fontweight="bold", pad=10)

    head = (f"Claude as judge at matched size: {win} wins {max(g,b)}-{min(g,b)}"
            if win != "tie" else f"Claude as judge at matched size: {g}-{b}, no decisive edge")
    suptitle(fig, head,
             f"Each pair judged twice (positions swapped); consistency {cons:.0%}, binomial p = {p:.1e}.")
    legend(fig)
    fig.subplots_adjust(top=0.76, bottom=0.12, wspace=0.24, left=0.07, right=0.97)
    fig.savefig(os.path.join(REPORTS, "eval_judge.png")); plt.close(fig)
    print("wrote eval_judge.png")

# ============================================================ scorecard
def fig_scorecard():
    j = R["judge"]; g, b, win, p, cons = dec_score(R)
    fig, ax = plt.subplots(figsize=(13.0, 5.0)); ax.set_xlim(0,13); ax.set_ylim(0,6); ax.axis("off")
    def tile(x, big, small, accent, big_fs=20):
        w, h = 2.85, 2.5
        ax.add_patch(FancyBboxPatch((x, 2.2), w, h, boxstyle="round,pad=0.02,rounding_size=0.16",
                     fc=PANEL, ec=accent, lw=2.0))
        ax.text(x+w/2, 2.2+h*0.60, big, ha="center", va="center", fontsize=big_fs, fontweight="bold", color=accent)
        ax.text(x+w/2, 2.2+h*0.21, small, ha="center", va="center", fontsize=9.0, color=SUB, linespacing=1.3)
    ppl_win = "BDH" if R["perplexity"]["bdh"] < R["perplexity"]["gpt"] else "GPT"
    tile(0.3,  f"{R['perplexity']['gpt']:.2f} / {R['perplexity']['bdh']:.2f}",
         f"perplexity GPT / BDH\n(lower better; {ppl_win} ahead)", BDHC if ppl_win=="BDH" else GPTC, 19)
    tile(3.35, f"{max(g,b)}-{min(g,b)}",
         f"decisive judge wins ({win})\np = {p:.0e}", WIN if win!="tie" else SUB, 22)
    tile(6.4,  f"{cons:.0%}", "position-swap consistency\n(controls judge bias)", GPTC, 22)
    tile(9.45, "matched", f"GPT {GP/1e3:.0f}K vs BDH {BP/1e3:.0f}K params\narchitecture, not size", WIN, 20)
    fig.text(0.5, 0.95, "Parameter-matched face-off: same size, same training, only the architecture differs",
             ha="center", va="top", fontsize=14.5, fontweight="bold", color=INK)
    fig.text(0.5, 0.9, f"Perplexity, real-word rate, and a swap-controlled Claude judge (n={j['n_judged']}) "
             "on identical prompts. BDH is marginally smaller, so it holds no size advantage.",
             ha="center", va="top", fontsize=9.6, color=SUB)
    fig.text(0.5, 0.06, "Base models are completion models (no answer-correctness); metrics measure how well each continues Shakespeare.",
             ha="center", va="bottom", fontsize=8.6, color=SUB, style="italic")
    fig.savefig(os.path.join(REPORTS, "eval_scorecard.png")); plt.close(fig)
    print("wrote eval_scorecard.png")

# ============================================================ matched vs confound
def fig_matched_vs_confound():
    if BIG is None:
        print("skip eval_matched_vs_confound.png (no base_eval_2x.json)"); return
    g0, b0, w0, p0, _ = dec_score(BIG)   # 2x
    g1, b1, w1, p1, _ = dec_score(R)     # matched
    bigBP = BIG["params"]["bdh"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.6, 5.6), sharey=True)
    for ax, (g, b, w, p, lab) in zip(
        (ax1, ax2),
        [(g0, b0, w0, p0, f"BDH at {bigBP/1e6:.2f}M  (~2x GPT)"),
         (g1, b1, w1, p1, f"BDH at {BP/1e3:.0f}K  (matched to GPT)")]):
        style(ax)
        bars = ax.bar([0, 1], [g, b], 0.55, color=[GPTC, BDHC])
        ax.set_xticks([0, 1]); ax.set_xticklabels([f"GPT\n{GP/1e3:.0f}K", "BDH"])
        for bb in bars:
            ax.annotate(f"{bb.get_height():.0f}", (bb.get_x()+bb.get_width()/2, bb.get_height()+0.3),
                        ha="center", va="bottom", fontsize=13, fontweight="bold")
        ax.set_title(lab, fontsize=12.5, fontweight="bold", pad=10)
        ax.set_xlabel(f"decisive judge wins  (p = {p:.1e})")
    ax1.set_ylabel("prompts won (both orderings)")
    ax1.set_ylim(0, max(g0,b0,g1,b1)*1.25)
    suptitle(fig, "Does the win survive parameter matching?",
             "Left: the size-confounded result. Right: same harness with the parameter gap removed.")
    fig.subplots_adjust(top=0.80, bottom=0.13, wspace=0.08, left=0.08, right=0.97)
    fig.savefig(os.path.join(REPORTS, "eval_matched_vs_confound.png")); plt.close(fig)
    print("wrote eval_matched_vs_confound.png")

if __name__ == "__main__":
    fig_objective(); fig_judge(); fig_scorecard(); fig_matched_vs_confound()
    print("done ->", REPORTS)
