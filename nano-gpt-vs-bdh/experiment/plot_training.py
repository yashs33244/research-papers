"""Base-pretraining curve for the paper, parsed from the training logs.

  reports/training_curves.png - validation loss vs iteration, matched GPT vs
                                matched BDH (both from scratch, identical settings).
"""
import os
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_THIS = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(_THIS, "out")
REPORTS = os.path.join(_THIS, "reports")

INK="#1A1D23"; SUB="#5A6472"; GRID="#E6E9EE"; GPTC="#2E86AB"; BDHC="#E9A12E"
plt.rcParams.update({"font.family":"Helvetica","axes.edgecolor":SUB,"axes.linewidth":0.9,
    "text.color":INK,"xtick.color":INK,"ytick.color":INK,"savefig.dpi":170,"savefig.bbox":"tight"})

def parse_base(path):
    it, val = [], []
    for line in open(path):
        m = re.search(r"iter\s+(\d+)\s+\|\s+train\s+([\d.]+)\s+\|\s+val\s+([\d.]+)", line)
        if m:
            it.append(int(m.group(1))); val.append(float(m.group(3)))
    return it, val

def style(ax):
    for s in ("top","right"): ax.spines[s].set_visible(False)
    ax.grid(color=GRID, lw=1); ax.set_axisbelow(True); ax.tick_params(length=0)

def main():
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    gi, gv = parse_base(os.path.join(OUT, "train_gpt.log"))
    bi, bv = parse_base(os.path.join(OUT, "train_bdh.log"))
    style(ax)
    ax.plot(gi, gv, "-o", color=GPTC, lw=2, ms=5, label=f"GPT  (best val {min(gv):.2f})")
    ax.plot(bi, bv, "-o", color=BDHC, lw=2, ms=5, label=f"BDH  (best val {min(bv):.2f})")
    ax.set_xlabel("training iteration"); ax.set_ylabel("validation loss (cross-entropy)")
    ax.legend(frameon=False, fontsize=11)
    fig.text(0.5, 0.99, "Base pretraining at matched size: both models learn",
             ha="center", va="top", fontsize=14.5, fontweight="bold", color=INK)
    fig.text(0.5, 0.945, "Next-character prediction on character-level TinyShakespeare; identical training settings, only the architecture differs.",
             ha="center", va="top", fontsize=9.2, color=SUB)
    fig.subplots_adjust(top=0.86, bottom=0.11, left=0.10, right=0.96)
    fig.savefig(os.path.join(REPORTS, "training_curves.png")); plt.close(fig)
    print(f"wrote training_curves.png  (GPT best {min(gv):.2f}, BDH best {min(bv):.2f})")

if __name__ == "__main__":
    main()
