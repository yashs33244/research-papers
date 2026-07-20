"""Download TinyShakespeare, build the char vocab, encode it, save train/val splits.

BEGINNER FIRST
--------------
Before we can train anything we need data in a shape the model likes. This
script does four things, once, up front:

  1. DOWNLOAD  the raw text (all of Shakespeare, concatenated) - about 1 MB.
  2. VOCAB     figure out every distinct character and give each one a number.
  3. ENCODE    rewrite the entire text as one long list of those numbers,
               then store it as a PyTorch tensor (a big 1-D array of ints).
  4. SPLIT     keep the first 90% for TRAINING and the last 10% for VALIDATION.
               We never train on the val slice; it is our honesty check that the
               model is actually learning language and not just memorizing.

Run it once:
    python data/prepare.py

It writes into the data/ folder:
    input.txt   the raw corpus (cached so we do not re-download)
    meta.json   the vocabulary (via CharTokenizer.save) - needed to decode later
    train.pt    training token ids as a torch.long tensor
    val.pt      validation token ids as a torch.long tensor

DEEPER DIVE
-----------
Why one giant tensor instead of separate "sentences"? Because a character LM is
trained on random windows sliced out of the stream (see get_batch in train.py).
There is no notion of a sentence boundary at the character level; the model just
learns "given these T characters, what comes next?". Storing the corpus as a
single contiguous tensor makes slicing those windows trivial and fast.
"""

from __future__ import annotations

import os
import sys
import urllib.request

import torch

# Make `import nanobdh` work whether you run this from the repo root or elsewhere.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from nanobdh.tokenizer import CharTokenizer  # noqa: E402

# Karpathy's canonical char-demo copy of the TinyShakespeare corpus.
DATA_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/"
    "tinyshakespeare/input.txt"
)

INPUT_PATH = os.path.join(_THIS_DIR, "input.txt")
META_PATH = os.path.join(_THIS_DIR, "meta.json")
TRAIN_PATH = os.path.join(_THIS_DIR, "train.pt")
VAL_PATH = os.path.join(_THIS_DIR, "val.pt")


def download() -> str:
    """Fetch the corpus once and cache it on disk; return the raw text."""
    if not os.path.exists(INPUT_PATH):
        print(f"downloading TinyShakespeare -> {INPUT_PATH}")
        urllib.request.urlretrieve(DATA_URL, INPUT_PATH)
    else:
        print(f"using cached corpus at {INPUT_PATH}")
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        return f.read()


def main() -> None:
    text = download()
    print(f"corpus length: {len(text):,} characters")

    # Step 2: build the vocabulary directly from the text.
    tok = CharTokenizer.from_text(text)
    print(f"vocab size (V): {tok.vocab_size}")
    print("vocab:", "".join(tok.chars).replace("\n", "\\n"))

    # Step 3: encode the whole corpus into one long tensor of token ids.
    # dtype=long because embedding layers and cross-entropy expect int64 indices.
    ids = torch.tensor(tok.encode(text), dtype=torch.long)

    # Step 4: 90/10 train/val split (the val slice is held out, never trained on).
    n = int(0.9 * len(ids))
    train_ids, val_ids = ids[:n], ids[n:]
    print(f"train tokens: {len(train_ids):,}   val tokens: {len(val_ids):,}")

    # Persist everything the training/sampling scripts will need.
    tok.save(META_PATH)
    torch.save(train_ids, TRAIN_PATH)
    torch.save(val_ids, VAL_PATH)
    print(f"wrote {META_PATH}, {TRAIN_PATH}, {VAL_PATH}")
    print("done. next: python -m nanobdh.train --model gpt")


if __name__ == "__main__":
    main()
