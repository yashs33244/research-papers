"""Post-training (SFT) for nano-bdh: turn a base char-LM into a chat "assistant".

BEGINNER FIRST
--------------
"Fine-tuning" means: take a model that already learned SOMETHING, and keep training
it on a NEW, smaller dataset so it picks up a new behavior without forgetting how to
form characters. Here the base model learned to write Shakespeare; we now show it
short conversations so it learns the TURN-TAKING FORMAT:

    User: <question>
    Assistant: <answer>

The mechanics are almost identical to nanobdh/train.py, with ONE crucial change:
we do SUPERVISED FINE-TUNING WITH LOSS MASKING. We only grade the model on the
characters of the ASSISTANT'S ANSWER, never on the user's question or the literal
"Assistant: " label. data/prepare_chat.py already built the mask for us: it saved a
`targets` tensor where every non-answer position is -1. We feed that -1 straight into
F.cross_entropy(..., ignore_index=-1), which makes those positions contribute zero
gradient. So the optimizer only ever learns "produce the answer, then stop."

HONEST EXPECTATIONS
-------------------
The base is ~1-2M params, character-level, trained on ~1MB of Shakespeare. After SFT
the model will reliably produce the conversation SHAPE (an "Assistant:" turn that
ends and hands back to "User:"), but the words themselves will be mostly broken
English. This file is a correct demonstration of the SFT METHOD, not a path to a
capable assistant.

WHY WE CALL model(idx) AND COMPUTE LOSS OURSELVES
-------------------------------------------------
The model's forward computes an UNMASKED cross-entropy when you pass targets. For
masked SFT we need ignore_index=-1, so we call model(idx) to get raw logits and then
compute F.cross_entropy(logits.view(-1, V), targets.view(-1), ignore_index=-1)
ourselves. Same math, just with the mask honored.

Run:
    python -m nanobdh.finetune --model gpt --max_iters 800
    python -m nanobdh.finetune --model bdh --max_iters 800
Loads out/<model>.pt (base) and saves out/<model>-chat.pt (chat).
"""

from __future__ import annotations

import argparse
import os

import torch
import torch.nn.functional as F

from .tokenizer import CharTokenizer
from .model_gpt import GPT, GPTConfig
from .model_bdh import BDH, BDHConfig

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
DATA_DIR = os.path.join(_REPO_ROOT, "data")
OUT_DIR = os.path.join(_REPO_ROOT, "out")
CHAT_DATA_PATH = os.path.join(DATA_DIR, "chat.pt")


def pick_device(requested: str) -> str:
    """Choose the compute device. 'auto' prefers MPS (Apple GPU), then CUDA, else CPU."""
    if requested != "auto":
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def rebuild_model(model_name: str, config: dict):
    """Reconstruct the right architecture from the saved base-checkpoint config dict.

    Using the base's own config guarantees the fine-tuned model is byte-for-byte the
    same shape, so load_state_dict fits perfectly and no weights are re-initialized.
    """
    if model_name == "gpt":
        return GPT(GPTConfig(**config))
    if model_name == "bdh":
        return BDH(BDHConfig(**config))
    raise ValueError(f"unknown model {model_name!r} (expected 'gpt' or 'bdh')")


def load_chat_data():
    """Load the masked SFT stream produced by data/prepare_chat.py.

    Returns two 1-D LongTensors of equal length: `tokens` (the rendered
    conversations concatenated) and `targets` (next-char labels with -1 on every
    non-assistant position). We keep them as one long stream and slice random
    windows, exactly like train.py slices the Shakespeare stream.
    """
    if not os.path.exists(CHAT_DATA_PATH):
        raise FileNotFoundError(
            f"missing {CHAT_DATA_PATH}. Run  python data/prepare_chat.py  first."
        )
    blob = torch.load(CHAT_DATA_PATH)
    tokens = blob["tokens"]
    targets = blob["targets"]
    assert tokens.shape == targets.shape, "tokens and targets must be the same length"
    return tokens, targets


def get_batch(tokens, targets, block_size, batch_size, device):
    """Sample `batch_size` random contiguous windows of length `block_size`.

    Unlike base training, we do NOT re-derive targets by shifting the input here.
    prepare_chat.py already aligned targets to tokens (target[i] is the label for
    position i, already shifted and already masked with -1). So we slice the SAME
    [i : i+T] window out of BOTH streams and keep them aligned.

    Consequence of slicing a concatenated stream: a window can straddle the boundary
    between two conversations. That is fine and even helpful - the model still only
    ever gets gradient on answer characters (everything else is -1), and it learns
    that after a turn ends ("\n\n...User:") a fresh turn can begin.
    """
    x_list, y_list = [], []
    max_start = len(tokens) - block_size
    ix = torch.randint(0, max_start, (batch_size,))
    for i in ix:
        i = int(i)
        x_list.append(tokens[i:i + block_size])
        y_list.append(targets[i:i + block_size])
    x = torch.stack(x_list).to(device)
    y = torch.stack(y_list).to(device)
    return x, y


def masked_loss(model, x, y):
    """Cross-entropy computed ONLY on answer positions (targets == -1 are ignored).

    We call model(x) to get raw logits (B, T, V), flatten to (B*T, V) and (B*T,),
    and let F.cross_entropy skip every -1 target. This is the beating heart of SFT.
    """
    logits, _ = model(x)                       # ignore the model's own unmasked loss
    V = logits.size(-1)
    loss = F.cross_entropy(
        logits.view(-1, V),
        y.view(-1),
        ignore_index=-1,                       # THE mask: -1 targets add zero gradient
    )
    return loss


@torch.no_grad()
def estimate_loss(model, tokens, targets, block_size, batch_size, eval_iters, device):
    """Average masked loss over a few batches with dropout off (a stable read-out).

    Note: if a sampled batch happens to contain ZERO supervised positions (all -1),
    cross_entropy returns NaN. We simply skip such rare batches so the average stays
    meaningful; with thousands of answer chars this almost never triggers.
    """
    model.eval()
    losses = []
    for _ in range(eval_iters):
        x, y = get_batch(tokens, targets, block_size, batch_size, device)
        if int((y != -1).sum().item()) == 0:
            continue
        losses.append(masked_loss(model, x, y).item())
    model.train()
    if not losses:
        return float("nan")
    return sum(losses) / len(losses)


def main():
    p = argparse.ArgumentParser(description="SFT post-training for nano-bdh (GPT or BDH).")
    p.add_argument("--model", choices=["gpt", "bdh"], required=True)
    p.add_argument("--device", default="auto", help="auto|mps|cuda|cpu")
    # Fine-tuning schedule. Fewer iters and a smaller LR than base training, because
    # we are ADAPTING an already-trained model, not learning language from scratch.
    p.add_argument("--max_iters", type=int, default=800)
    p.add_argument("--eval_interval", type=int, default=100)
    p.add_argument("--eval_iters", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-1)
    p.add_argument("--seed", type=int, default=1337)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    print(f"device: {device}")

    base_path = os.path.join(OUT_DIR, f"{args.model}.pt")
    if not os.path.exists(base_path):
        raise FileNotFoundError(
            f"no base checkpoint at {base_path}. Train the base first: "
            f"python -m nanobdh.train --model {args.model}"
        )

    # 1. Load the BASE checkpoint (config + weights + vocab) and rebuild the model.
    ckpt = torch.load(base_path, map_location=device)
    assert ckpt["model"] == args.model, (
        f"checkpoint says model={ckpt['model']!r} but you asked for {args.model!r}"
    )
    config = ckpt["config"]
    tok = CharTokenizer(ckpt["vocab"])  # EXACT base vocab - do not rebuild from data
    model = rebuild_model(args.model, config)
    model.load_state_dict(ckpt["state_dict"])   # start from the trained base weights
    model.to(device)
    model.train()
    block_size = config["block_size"]
    print(f"loaded base {args.model} | params: {model.num_params():,} | V={tok.vocab_size} | T={block_size}")

    # 2. Load the masked SFT data.
    tokens, targets = load_chat_data()
    supervised = int((targets != -1).sum().item())
    print(
        f"chat data: {len(tokens):,} chars, "
        f"{supervised:,} supervised ({100.0 * supervised / len(tokens):.1f}%)"
    )

    # 3. Optimizer. Same AdamW as base training but a gentler LR (see argparse note).
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"{args.model}-chat.pt")
    best = float("inf")

    # 4. The SFT loop: forward -> masked loss -> backward -> step, with periodic eval.
    for it in range(args.max_iters + 1):
        if it % args.eval_interval == 0 or it == args.max_iters:
            loss_est = estimate_loss(
                model, tokens, targets, block_size,
                args.batch_size, args.eval_iters, device,
            )
            print(f"iter {it:5d} | sft loss {loss_est:.4f}")
            # Save the best-so-far chat model, mirroring train.py's checkpoint dict
            # EXACTLY so serve.py / sample.py can load it with no special-casing.
            if loss_est == loss_est and loss_est < best:  # (loss==loss) rejects NaN
                best = loss_est
                torch.save(
                    {
                        "model": args.model,
                        "state_dict": model.state_dict(),
                        "config": config,       # same shape as the base
                        "vocab": tok.chars,      # same vocab as the base
                    },
                    out_path,
                )

        if it == args.max_iters:
            break

        x, y = get_batch(tokens, targets, block_size, args.batch_size, device)
        if int((y != -1).sum().item()) == 0:
            continue  # skip the rare all-masked batch (undefined loss)
        loss = masked_loss(model, x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    print(f"done. best sft loss {best:.4f}. chat checkpoint -> {out_path}")
    print(f"serve both models with:  python serve.py")


if __name__ == "__main__":
    main()
