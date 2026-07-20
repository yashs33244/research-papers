"""Reproduce BDH's interpretability "ahas" on a trained checkpoint.

This is the code behind Chapter 9. Right now it implements Aha #1:

  AHA 1 - SPARSE, POSITIVE ACTIVATIONS
  ------------------------------------
  BDH lifts each position into a big neuron space of size N and applies ReLU, so
  every neuron value is >= 0 and (the claim) only a few percent are non-zero at
  any moment. We verify that empirically: run a real chunk of Shakespeare through
  the trained model and measure, at every ReLU, what fraction of neurons fired.

  HOW WE MEASURE IT (no model surgery): model_bdh.py computes its sparse fields
  with `F.relu(...)`. We temporarily wrap `torch.nn.functional.relu` so that every
  time the model calls it, we record the fraction of outputs that came out > 0.
  Then we restore the original. This is a clean, read-only probe.

Run:
    python -m nanobdh.analyze --what sparsity
"""

from __future__ import annotations

import argparse
import os

import torch
import torch.nn.functional as F

from .tokenizer import CharTokenizer
from .model_bdh import BDH, BDHConfig
from .sample import pick_device, OUT_DIR

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
_INPUT = os.path.join(_REPO_ROOT, "data", "input.txt")
_REPORTS = os.path.join(_REPO_ROOT, "reports")


def load_bdh(device: str):
    """Load the trained BDH exactly the way sample.py does."""
    ckpt_path = os.path.join(OUT_DIR, "bdh.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"no checkpoint at {ckpt_path}. Train first: python -m nanobdh.train --model bdh"
        )
    ckpt = torch.load(ckpt_path, map_location=device)
    tok = CharTokenizer(ckpt["vocab"])
    model = BDH(BDHConfig(**ckpt["config"]))
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, tok, ckpt


def sample_text(n_chars: int) -> str:
    """Grab a chunk from the validation region (last 10%) of the corpus."""
    text = open(_INPUT, "r", encoding="utf-8").read()
    val_start = int(0.9 * len(text))
    return text[val_start : val_start + n_chars]


def measure_sparsity(model, tok, device, text: str):
    """Return (per_relu_active_fraction, overall_mean).

    We monkeypatch F.relu to log the active fraction of each call's output, run a
    single forward pass, then restore F.relu.
    """
    original_relu = F.relu
    active_fractions = []

    def spy(x, *args, **kwargs):
        y = original_relu(x, *args, **kwargs)
        active_fractions.append((y > 0).float().mean().item())
        return y

    F.relu = spy
    try:
        ids = torch.tensor([tok.encode(text)], dtype=torch.long, device=device)
        with torch.no_grad():
            model(ids)
    finally:
        F.relu = original_relu  # always restore, even if the forward pass errors

    overall = sum(active_fractions) / max(1, len(active_fractions))
    return active_fractions, overall


def plot_sparsity(active_fractions, overall, out_path):
    """Save a simple bar chart: active fraction at each ReLU call."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))
    xs = list(range(1, len(active_fractions) + 1))
    ax.bar(xs, [f * 100 for f in active_fractions], color="#2A9D8F")
    ax.axhline(overall * 100, color="#D1495B", ls="--",
               label=f"mean {overall*100:.1f}% active")
    ax.set_xlabel("ReLU call (through the layers, in order)")
    ax.set_ylabel("neurons active  (%)")
    ax.set_title("BDH activation sparsity on real text\n(lower = sparser; most neurons stay at 0)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print("wrote", out_path)


def main():
    p = argparse.ArgumentParser(description="Reproduce BDH interpretability ahas.")
    p.add_argument("--what", choices=["sparsity"], default="sparsity")
    p.add_argument("--device", default="auto", help="auto|mps|cuda|cpu")
    p.add_argument("--chars", type=int, default=256, help="how many chars of context to probe")
    p.add_argument("--no_plot", action="store_true")
    args = p.parse_args()

    device = pick_device(args.device)
    model, tok, _ = load_bdh(device)
    text = sample_text(args.chars)

    if args.what == "sparsity":
        fracs, overall = measure_sparsity(model, tok, device, text)
        print(f"device: {device}")
        print(f"probed {len(text)} characters of held-out Shakespeare")
        print(f"ReLU calls observed: {len(fracs)}")
        print(f"per-call active fraction: " + ", ".join(f"{f*100:.1f}%" for f in fracs))
        print(f"OVERALL mean active fraction: {overall*100:.2f}%  "
              f"(so ~{100-overall*100:.0f}% of neurons are exactly 0)")
        if not args.no_plot:
            os.makedirs(_REPORTS, exist_ok=True)
            plot_sparsity(fracs, overall, os.path.join(_REPORTS, "bdh_sparsity.png"))


if __name__ == "__main__":
    main()
