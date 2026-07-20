"""Character-level tokenizer.

BEGINNER FIRST
--------------
A neural network cannot read letters. It only does math on numbers. So the very
first job in any language model is to turn text into numbers and back again.
That is ALL a "tokenizer" is: a two-way dictionary.

We use the simplest possible scheme: one character = one number. The string
"cat" becomes something like [12, 40, 58]. There is no cleverness, no
sub-word merging (that would be "BPE", used by real GPT-2). We deliberately
avoid BPE here because characters are the easiest thing to reason about while
learning, and TinyShakespeare only has about 65 distinct characters anyway.

The two directions:
    encode("hi") -> [46, 47]      (string  -> list of ints, for feeding the model)
    decode([46, 47]) -> "hi"      (list of ints -> string, for reading its output)

DEEPER DIVE
-----------
"stoi" = string-to-integer (char -> id). "itos" = integer-to-string (id -> char).
The vocabulary is just the sorted set of unique characters in the training text,
so the mapping is fully determined by the data and is 100% reversible (no
unknown-token problem, because every char in the corpus is in the vocab).

The exact same vocabulary MUST be reused at generation time, otherwise id 12
might mean a different character than it did during training. We therefore save
the vocab to disk (see data/prepare.py) and reload it in sample.py.
"""

from __future__ import annotations

import json
from typing import List


class CharTokenizer:
    """A reversible character <-> integer mapping built from a corpus."""

    def __init__(self, chars: List[str]):
        # `chars` is the ordered vocabulary: index i in this list is token id i.
        # Keeping it sorted makes the mapping deterministic and reproducible.
        self.chars = list(chars)
        self.vocab_size = len(self.chars)

        # The two lookup tables that do the actual work.
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}  # char -> id
        self.itos = {i: ch for i, ch in enumerate(self.chars)}  # id   -> char

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        """Build a vocabulary by scanning every character in `text`.

        `set(text)` collapses the whole corpus down to its unique characters;
        `sorted(...)` gives them a stable, reproducible order.
        """
        chars = sorted(set(text))
        return cls(chars)

    def encode(self, s: str) -> List[int]:
        """Text -> list of token ids. This is what we feed the model."""
        return [self.stoi[c] for c in s]

    def decode(self, ids: List[int]) -> str:
        """List of token ids -> text. This is how we read the model's output."""
        return "".join(self.itos[int(i)] for i in ids)

    # ------------------------------------------------------------------
    # Persistence: the vocab used to TRAIN must be the vocab used to SAMPLE.
    # We store it as plain JSON so it is human-inspectable and portable.
    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"chars": self.chars}, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "CharTokenizer":
        with open(path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        return cls(meta["chars"])
