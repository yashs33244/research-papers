"""Generate text from a trained checkpoint (works for either model).

BEGINNER FIRST
--------------
"Sampling" = asking the trained model to write. We give it a short PROMPT (or
just a newline), and it produces one character at a time, each time feeding its
own output back in as the new context. Two knobs shape the style:

  --temperature : how adventurous the model is.
      Low (0.5)  -> plays it safe, picks the most likely characters, repetitive.
      1.0        -> its natural distribution.
      High (1.3) -> takes more risks, more creative, more typos.
  --top_k       : only ever sample from the K most likely next characters, which
      prunes the long tail of nonsense. Smaller K = tighter, safer text.

Run (after training):
    python -m nanobdh.sample --model gpt --prompt "ROMEO:" --max_new_tokens 500
    python -m nanobdh.sample --model bdh --temperature 0.8 --top_k 40

DEEPER DIVE
-----------
We reload the SAME vocabulary that was saved in the checkpoint so token ids map
back to the exact characters the model trained on. The model's own generate()
method (see model_gpt.py / model_bdh.py) implements the crop -> logits ->
temperature -> top-k -> multinomial-sample loop; this script just wires up the
device, the tokenizer, and the prompt.
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
OUT_DIR = os.path.join(_REPO_ROOT, "out")


def pick_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def rebuild_model(model_name: str, config: dict):
    """Reconstruct the right architecture from the saved config dict."""
    if model_name == "gpt":
        return GPT(GPTConfig(**config))
    elif model_name == "bdh":
        return BDH(BDHConfig(**config))
    raise ValueError(f"unknown model {model_name!r}")


def main():
    p = argparse.ArgumentParser(description="Sample from a trained nano-bdh model.")
    p.add_argument("--model", choices=["gpt", "bdh"], required=True)
    p.add_argument("--ckpt", default=None, help="checkpoint path (default out/<model>.pt)")
    p.add_argument("--device", default="auto", help="auto|mps|cuda|cpu")
    p.add_argument("--prompt", default="\n", help="text to condition on")
    p.add_argument("--max_new_tokens", type=int, default=500)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=40)
    p.add_argument("--seed", type=int, default=1337)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = pick_device(args.device)

    ckpt_path = args.ckpt or os.path.join(OUT_DIR, f"{args.model}.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"no checkpoint at {ckpt_path}. Train first: python -m nanobdh.train --model {args.model}"
        )

    ckpt = torch.load(ckpt_path, map_location=device)
    tok = CharTokenizer(ckpt["vocab"])  # exact vocab used during training
    model = rebuild_model(ckpt["model"], ckpt["config"])
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()

    # Encode the prompt into a (1, T) batch of token ids.
    start_ids = tok.encode(args.prompt)
    idx = torch.tensor([start_ids], dtype=torch.long, device=device)

    out = model.generate(
        idx,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    print(tok.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
