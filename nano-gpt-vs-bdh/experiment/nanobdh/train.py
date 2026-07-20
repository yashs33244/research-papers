"""Shared training loop for BOTH models. Pick one with --model gpt|bdh.

BEGINNER FIRST
--------------
Training a language model is a loop of four steps, repeated thousands of times:

  1. GRAB a batch: cut B random windows of length T out of the training text.
     For each window the "input" is characters 0..T-1 and the "target" is the
     SAME window shifted by one (characters 1..T) - i.e. "the next char" at every
     position. That is the whole supervision signal.
  2. FORWARD: run the model, get its predicted probabilities, and measure how
     wrong it is with cross-entropy LOSS (lower = better predictions).
  3. BACKWARD: autograd computes how to nudge every weight to reduce the loss.
  4. STEP: the AdamW optimizer applies those nudges.

Every so often we pause and estimate the loss on BOTH the train and the held-out
val split (with gradients off) so we can watch the model actually learn.

Run:
    python -m nanobdh.train --model gpt
    python -m nanobdh.train --model bdh --max_iters 3000

DEEPER DIVE
-----------
- DEVICE: we prefer Apple's MPS GPU, then CUDA, else CPU. Apple's MPS backend
  does not implement every op; if something is missing you can force CPU with
  --device cpu.
- AdamW is Adam with decoupled weight decay - the standard optimizer for
  Transformers. Cross-entropy is the standard next-token loss.
- estimate_loss() averages over several batches with the model in eval() mode so
  dropout is disabled and the number is stable. The gap between train and val
  loss is your overfitting gauge.
"""

from __future__ import annotations

import argparse
import os

import torch

from .tokenizer import CharTokenizer
from .model_gpt import GPT, GPTConfig
from .model_bdh import BDH, BDHConfig

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
DATA_DIR = os.path.join(_REPO_ROOT, "data")
OUT_DIR = os.path.join(_REPO_ROOT, "out")


def pick_device(requested: str) -> str:
    """Choose the compute device. 'auto' prefers MPS (Apple GPU), then CUDA, else CPU."""
    if requested != "auto":
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_data():
    """Load the token tensors and vocab produced by data/prepare.py."""
    meta_path = os.path.join(DATA_DIR, "meta.json")
    train_path = os.path.join(DATA_DIR, "train.pt")
    val_path = os.path.join(DATA_DIR, "val.pt")
    if not (os.path.exists(meta_path) and os.path.exists(train_path)):
        raise FileNotFoundError(
            "data not found. Run:  python data/prepare.py  first."
        )
    tok = CharTokenizer.load(meta_path)
    train_ids = torch.load(train_path)
    val_ids = torch.load(val_path)
    return tok, train_ids, val_ids


def get_batch(data: torch.Tensor, block_size: int, batch_size: int, device: str):
    """Sample `batch_size` random windows of length `block_size`.

    x = the window; y = the same window shifted right by one (the next char at
    each position). Both come back as (B, T) long tensors on `device`.
    """
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + 1 + block_size] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, train_ids, val_ids, block_size, batch_size, eval_iters, device):
    """Average loss over `eval_iters` batches for each split, dropout off."""
    out = {}
    model.eval()
    for split, data in (("train", train_ids), ("val", val_ids)):
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(data, block_size, batch_size, device)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def build_model(model_name: str, vocab_size: int, block_size: int, args):
    """Construct GPT or BDH with a shared set of size knobs from argparse."""
    if model_name == "gpt":
        cfg = GPTConfig(
            vocab_size=vocab_size,
            block_size=block_size,
            n_layer=args.n_layer,
            n_head=args.n_head,
            n_embd=args.n_embd,
            dropout=args.dropout,
        )
        return GPT(cfg), cfg
    elif model_name == "bdh":
        cfg = BDHConfig(
            vocab_size=vocab_size,
            block_size=block_size,
            n_layer=args.n_layer,
            n_head=args.n_head,
            n_embd=args.n_embd,
            dropout=args.dropout,
            neuron_dim_multiplier=args.neuron_dim_multiplier,
        )
        return BDH(cfg), cfg
    raise ValueError(f"unknown model {model_name!r} (expected 'gpt' or 'bdh')")


def main():
    p = argparse.ArgumentParser(description="Train nano-bdh (GPT or BDH).")
    p.add_argument("--model", choices=["gpt", "bdh"], required=True)
    p.add_argument("--device", default="auto", help="auto|mps|cuda|cpu")
    # Training schedule
    p.add_argument("--max_iters", type=int, default=3000)
    p.add_argument("--eval_interval", type=int, default=250)
    p.add_argument("--eval_iters", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--block_size", type=int, default=128)  # T
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-1)
    p.add_argument("--seed", type=int, default=1337)
    # Model size knobs (shared names across both architectures)
    p.add_argument("--n_layer", type=int, default=4)
    p.add_argument("--n_head", type=int, default=4)
    p.add_argument("--n_embd", type=int, default=128)  # C
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--neuron_dim_multiplier", type=int, default=32,
                   help="BDH only: neuron-space size relative to C")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    print(f"device: {device}")

    tok, train_ids, val_ids = load_data()
    print(f"vocab size (V): {tok.vocab_size}")

    model, cfg = build_model(args.model, tok.vocab_size, args.block_size, args)
    model.to(device)
    print(f"model: {args.model}  params: {model.num_params():,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    os.makedirs(OUT_DIR, exist_ok=True)
    ckpt_path = os.path.join(OUT_DIR, f"{args.model}.pt")
    best_val = float("inf")

    for it in range(args.max_iters + 1):
        # Periodic evaluation + checkpointing of the best model so far.
        if it % args.eval_interval == 0 or it == args.max_iters:
            losses = estimate_loss(
                model, train_ids, val_ids,
                args.block_size, args.batch_size, args.eval_iters, device,
            )
            print(f"iter {it:5d} | train {losses['train']:.4f} | val {losses['val']:.4f}")
            if losses["val"] < best_val:
                best_val = losses["val"]
                torch.save(
                    {
                        "model": args.model,
                        "state_dict": model.state_dict(),
                        "config": vars(cfg),
                        "vocab": tok.chars,
                    },
                    ckpt_path,
                )

        if it == args.max_iters:
            break

        # One optimization step: forward -> loss -> backward -> update.
        x, y = get_batch(train_ids, args.block_size, args.batch_size, device)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    print(f"done. best val loss {best_val:.4f}. checkpoint -> {ckpt_path}")
    print(f"sample with:  python -m nanobdh.sample --model {args.model}")


if __name__ == "__main__":
    main()
