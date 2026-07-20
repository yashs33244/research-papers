"""nano-bdh: a from-scratch, teaching-first build of two small language models.

Two models live side by side in this package and share ONE training loop:

  1. GPT  (nanobdh.model_gpt)  - a classic GPT-2-style Transformer. This is the
                                 "baseline" every modern LLM descends from.
  2. BDH  (nanobdh.model_bdh)  - the "Dragon Hatchling", a brain-inspired
                                 alternative (arXiv:2509.26507). Different math,
                                 same job: predict the next character.

Shared notation used EVERYWHERE in this codebase (memorize these six letters):
    B      = batch size          (how many sequences we process at once)
    T      = block/context length (how many characters the model looks back on)
    C      = embedding dimension  (the size of each token's "meaning" vector)
    V      = vocab size           (how many distinct characters exist, ~65 here)
    n_head = number of attention heads
    n_layer= number of stacked blocks (depth of the model)

Both models expose the SAME three methods so train.py / sample.py do not care
which one they were handed:
    model(idx, targets=None) -> (logits, loss)
    model.generate(idx, max_new_tokens, temperature, top_k) -> idx
"""

from .tokenizer import CharTokenizer
from .model_gpt import GPT, GPTConfig
from .model_bdh import BDH, BDHConfig

__all__ = ["CharTokenizer", "GPT", "GPTConfig", "BDH", "BDHConfig"]
