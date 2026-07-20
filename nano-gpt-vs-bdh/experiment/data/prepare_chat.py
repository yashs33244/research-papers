"""Build the SFT (post-training) dataset that turns a base char-LM into a chat model.

BEGINNER FIRST
--------------
The base models (out/gpt.pt, out/bdh.pt) only learned one thing: "given the last
few characters of Shakespeare, predict the next character." They have never seen a
conversation. POST-TRAINING (here, plain Supervised Fine-Tuning / SFT) teaches the
CONVERSATION FORMAT: it shows the model many short examples written as

    User: <question>
    Assistant: <answer>

and trains it to continue an "Assistant:" prefix with a helpful-looking reply, then
stop. Two honest caveats you must keep in mind:

  1. The base is tiny (~1-2M params), character-level, trained on ~1MB of
     Shakespeare. SFT can teach the TURN-TAKING SHAPE, but the actual words will be
     mostly broken English. This is a METHOD demonstration, not a capable assistant.

  2. We reuse the EXACT base vocabulary (data/meta.json, ~65 characters). We do NOT
     add special "<user>"/"<assistant>" tokens, because that would resize the
     embedding matrix and make the base checkpoint impossible to load. The role
     markers ("User:", "Assistant:") are just PLAIN TEXT made of characters that
     already exist in the vocab.

THE ONE IDEA THAT MAKES SFT WORK: LOSS MASKING
----------------------------------------------
We only want the model to LEARN to produce the assistant's answer. We do NOT want it
to waste capacity learning to generate the human's questions, nor the literal
"Assistant: " label (those are given to it at chat time, not predicted). So for every
training example we build two equal-length tensors:

    tokens  : the full rendered conversation, as token ids.
    targets : the SAME sequence shifted by one (next-char labels), EXCEPT every
              position that is not part of the assistant's answer is set to -1.

At loss time we call F.cross_entropy(..., ignore_index=-1), so the -1 positions
contribute ZERO gradient. The model is graded only on the characters of the answer
(including the trailing blank line that ends the turn). See nanobdh/finetune.py.

WHAT COUNTS AS "THE ANSWER" (must match finetune.py and serve.py exactly)
-------------------------------------------------------------------------
One example is rendered EXACTLY as (contract 2):

    "User: <question>\nAssistant: <answer>\n\n"

The supervised region is <answer> plus the final "\n\n" that terminates the turn.
Everything else ("User: <question>\n" and the literal "Assistant: " prefix) is
masked to -1. Because targets are the NEXT character, the mask is applied on the
target axis: a target position is kept only if the character it PREDICTS lies inside
the answer region.

Run:
    python data/prepare_chat.py
It writes data/chat.pt = {"tokens": LongTensor, "targets": LongTensor}, both 1-D and
the same length, ready for nanobdh/finetune.py to slice into batches.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request

import torch

# Make `import nanobdh` work no matter where this is run from.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from nanobdh.tokenizer import CharTokenizer  # noqa: E402

META_PATH = os.path.join(_THIS_DIR, "meta.json")
CHAT_PATH = os.path.join(_THIS_DIR, "chat.pt")

# A tiny, well-known instruction dataset served as raw JSON over https. Alpaca-style
# entries look like {"instruction": ..., "input": ..., "output": ...}. We only need
# short, simple everyday-English pairs, so we filter hard after download. If the
# download fails (offline, URL moved, etc.) we FALL BACK to a built-in synthetic set
# so this script ALWAYS produces data.
REMOTE_URL = (
    "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/"
    "alpaca_data.json"
)

# How long a single rendered example may get, in characters, AFTER cleaning. We keep
# examples short because the base block_size is 128; very long turns would be cropped
# mid-answer during generation and teach nothing useful. This also biases the set
# toward simple, learnable Q/A.
MAX_QUESTION_CHARS = 64
MAX_ANSWER_CHARS = 96
# Cap the total number of examples so finetuning stays fast on a laptop.
MAX_EXAMPLES = 4000


# --------------------------------------------------------------------------------
# Cleaning: force every example down to ONLY characters present in the base vocab.
# --------------------------------------------------------------------------------
def build_char_filter(vocab_chars):
    """Return a function that maps arbitrary text to text using only in-vocab chars.

    Strategy (simple and predictable):
      - A few common out-of-vocab characters get a sensible in-vocab REPLACEMENT
        (curly quotes -> straight quote, en/em dash -> hyphen, etc.). Note our vocab
        has NO straight double-quote either, so double quotes are dropped.
      - Any remaining character not in the vocab is DROPPED.
    We deliberately do NOT lowercase: the vocab contains both cases, and keeping
    "User:"/"Assistant:" capitalized matters for the format. (Lowercasing would be
    acceptable per the contract, but preserving case keeps the markers crisp.)
    """
    vocab = set(vocab_chars)

    # Map a handful of frequent unicode/punctuation cases onto in-vocab equivalents.
    # Anything whose replacement is "" is effectively deleted. The "smart" unicode
    # keys are written via chr(codepoint) so this source file itself contains no
    # fancy punctuation characters (only plain ASCII).
    replacements = {
        chr(0x2019): "'",   # right single quote -> apostrophe
        chr(0x2018): "'",   # left single quote  -> apostrophe
        chr(0x201C): "",    # left double quote  -> drop (no ASCII '"' in vocab)
        chr(0x201D): "",    # right double quote -> drop
        '"': "",            # straight double quote -> drop (not in vocab)
        chr(0x2014): "-",   # em dash  -> hyphen
        chr(0x2013): "-",   # en dash  -> hyphen
        chr(0x2026): "...", # ellipsis -> three dots
        "\t": " ",          # tab -> space
        "\r": "",           # carriage return -> drop
        "(": "",            # parens not in vocab -> drop
        ")": "",
        "[": "",
        "]": "",
        "/": " ",           # slash -> space (keeps word boundaries)
        "*": "",
        "#": "",
        "%": "",
        "+": "",
        "=": "",
        "_": " ",
    }

    def clean(text: str) -> str:
        out = []
        for ch in text:
            if ch in vocab:
                out.append(ch)
                continue
            rep = replacements.get(ch)
            if rep is None:
                # Unknown char with no explicit rule: drop it entirely.
                continue
            # Keep only the parts of the replacement that are actually in-vocab.
            for rc in rep:
                if rc in vocab:
                    out.append(rc)
        # Collapse any runs of spaces the cleaning may have introduced, and trim.
        cleaned = "".join(out)
        while "  " in cleaned:
            cleaned = cleaned.replace("  ", " ")
        return cleaned.strip()

    return clean


# --------------------------------------------------------------------------------
# Data sources.
# --------------------------------------------------------------------------------
def try_download_pairs():
    """Attempt to fetch a public instruction set. Return a list of (q, a) or None.

    We keep ONLY entries that have no separate "input" field (so the instruction is
    self-contained) and that are short. Everything is returned raw here; cleaning and
    length limits are applied later in build_examples so both remote and synthetic
    data go through the identical pipeline.
    """
    try:
        print(f"trying to download instruction data from:\n  {REMOTE_URL}")
        req = urllib.request.Request(REMOTE_URL, headers={"User-Agent": "nano-bdh"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            raw = resp.read().decode("utf-8")
        data = json.loads(raw)
        pairs = []
        for row in data:
            instr = (row.get("instruction") or "").strip()
            inp = (row.get("input") or "").strip()
            out = (row.get("output") or "").strip()
            # Skip examples that need an extra input block; we want pure Q -> A.
            if inp:
                continue
            if not instr or not out:
                continue
            pairs.append((instr, out))
        print(f"  downloaded {len(pairs):,} candidate self-contained pairs")
        return pairs
    except Exception as e:  # noqa: BLE001 - any failure means: fall back offline.
        print(f"  download failed ({e.__class__.__name__}: {e}). Using synthetic set.")
        return None


def synthetic_pairs():
    """A built-in, offline set of simple everyday-English Q/A dialogues.

    These are deliberately short and use ONLY characters that are certainly in the
    Shakespeare vocab (letters, space, and . , ! ? ' : ; -). We template a handful of
    patterns over small word lists to reach a few hundred varied examples, so the
    script is fully functional with no network at all.

    The point is to teach FORMAT, not facts: repeated, clean, short turns give the
    tiny model the strongest possible signal for "after 'Assistant: ', produce a
    short answer then stop."
    """
    pairs = []

    # 1. Fixed hand-written conversational pairs (greetings, small talk, manners).
    fixed = [
        ("Hello.", "Hello! How are you today?"),
        ("Hi there.", "Hi! It is nice to meet you."),
        ("How are you?", "I am doing well, thank you. How about you?"),
        ("What is your name?", "I am a small assistant. You can just call me Assistant."),
        ("Who are you?", "I am a tiny language model here to chat with you."),
        ("Thank you.", "You are very welcome!"),
        ("Thanks a lot.", "My pleasure. I am happy to help."),
        ("Goodbye.", "Goodbye! Have a wonderful day."),
        ("See you later.", "See you later! Take care."),
        ("Good morning.", "Good morning! I hope you slept well."),
        ("Good night.", "Good night! Sleep well and rest."),
        ("Please help me.", "Of course. I am glad to help you."),
        ("Can you help me?", "Yes, I can. What do you need help with?"),
        ("Are you a robot?", "I am a small computer program that can chat."),
        ("What can you do?", "I can chat with you and answer simple questions."),
        ("How is the weather?", "I cannot see outside, but I hope it is sunny for you."),
        ("Tell me a joke.", "Why did the cat sit on the computer? To keep an eye on the mouse!"),
        ("I am tired.", "You should take a short rest. You will feel better soon."),
        ("I am happy.", "That is wonderful to hear! I am happy for you."),
        ("I am sad.", "I am sorry to hear that. I hope your day gets better."),
        ("What time is it?", "I am not sure of the time, but it is a good time to smile."),
        ("Do you like music?", "I think music is lovely. What music do you like?"),
        ("What is your favorite color?", "I like the color blue. It is calm and clear."),
        ("Are you smart?", "I am small and simple, but I try my best to help."),
        ("Nice to meet you.", "Nice to meet you too! How can I help today?"),
    ]
    pairs.extend(fixed)

    # 2. Templated pairs: fill small slot lists into patterns. Everything stays in
    #    the vocab because we only use plain letters and the punctuation above.
    animals = ["cat", "dog", "bird", "fish", "horse", "rabbit", "mouse", "sheep"]
    colors = ["red", "blue", "green", "yellow", "white", "black", "orange", "purple"]
    foods = ["bread", "apple", "cheese", "soup", "rice", "cake", "honey", "milk"]
    places = ["the park", "the market", "the river", "the garden", "the hall", "the field"]

    for a in animals:
        pairs.append((f"Do you like the {a}?", f"Yes, the {a} is a fine and gentle creature."))
        pairs.append((f"What sound does a {a} make?", f"The {a} makes its own happy little sound."))
        pairs.append((f"Tell me about a {a}.", f"A {a} is a small friend that likes to play and rest."))

    for c in colors:
        pairs.append((f"Do you like the color {c}?", f"Yes, {c} is a lovely and pleasant color."))
        pairs.append((f"What is {c}?", f"The color {c} is bright and easy on the eyes."))

    for fd in foods:
        pairs.append((f"Do you like {fd}?", f"I think {fd} is tasty and good to eat."))
        pairs.append((f"Tell me about {fd}.", f"Some {fd} is a simple and pleasant treat."))

    for pl in places:
        pairs.append((f"What is at {pl}?", f"At {pl} you can walk, rest, and enjoy the calm air."))
        pairs.append((f"Can we go to {pl}?", f"Yes, let us go to {pl} and enjoy the day."))

    return pairs


# --------------------------------------------------------------------------------
# Rendering + masking: turn (q, a) pairs into token/target tensors.
# --------------------------------------------------------------------------------
def build_examples(pairs, tok, clean):
    """Clean each (q, a), render to the chat format, and build masked target ids.

    For each surviving example we produce two equal-length lists:
        ex_tokens  : ids of "User: <q>\nAssistant: <a>\n\n"
        ex_targets : next-char labels, with -1 everywhere OUTSIDE the answer region.

    The answer region (what the model is graded on) is <a> plus the terminating
    "\n\n". We compute it precisely by character offsets so the mask is exact.
    """
    all_tokens = []
    all_targets = []
    kept = 0

    for q_raw, a_raw in pairs:
        q = clean(q_raw)
        a = clean(a_raw)
        if not q or not a:
            continue
        if len(q) > MAX_QUESTION_CHARS or len(a) > MAX_ANSWER_CHARS:
            continue

        # The exact rendering from contract 2. The PREFIX is everything the model is
        # GIVEN at chat time; the ANSWER is everything it must LEARN to produce.
        prefix = f"User: {q}\nAssistant: "
        answer = f"{a}\n\n"          # answer text plus the blank line that ends the turn
        full = prefix + answer

        ids = tok.encode(full)      # list[int], length L
        L = len(ids)
        prefix_len = len(prefix)    # chars in prefix == ids in prefix (char-level)

        # Build next-char targets: target[i] = ids[i+1]; last position has none (-1).
        # A target at position i is KEPT only if the char it predicts (ids[i+1]) is
        # inside the answer region, i.e. its index i+1 >= prefix_len. That means the
        # very first answer character (right after "Assistant: ") IS supervised, and
        # the "Assistant: " label itself is NOT (its own next-char predictions fall
        # before prefix_len).
        ex_tokens = ids
        ex_targets = []
        for i in range(L):
            nxt_index = i + 1
            if nxt_index >= L:
                ex_targets.append(-1)          # no next char for the final token
            elif nxt_index >= prefix_len:
                ex_targets.append(ids[nxt_index])  # supervised: predicting an answer char
            else:
                ex_targets.append(-1)          # masked: predicting a prefix char
        all_tokens.extend(ex_tokens)
        all_targets.extend(ex_targets)
        kept += 1
        if kept >= MAX_EXAMPLES:
            break

    return all_tokens, all_targets, kept


def main() -> None:
    if not os.path.exists(META_PATH):
        raise FileNotFoundError(
            f"missing {META_PATH}. Run  python data/prepare.py  first to create the "
            "base vocabulary (we must reuse the EXACT same vocab)."
        )

    # Reuse the EXACT base vocabulary. This is the whole reason SFT can load the base
    # checkpoint later: same vocab -> same embedding size -> same weights fit.
    tok = CharTokenizer.load(META_PATH)
    print(f"reusing base vocab: V = {tok.vocab_size} characters")
    clean = build_char_filter(tok.chars)

    # Get raw pairs (remote if possible, else the offline synthetic set).
    pairs = try_download_pairs()
    if not pairs:
        pairs = synthetic_pairs()
        source = "synthetic (offline fallback)"
    else:
        source = "downloaded instruction set"
    print(f"source: {source}  |  {len(pairs):,} raw pairs before cleaning")

    tokens, targets, kept = build_examples(pairs, tok, clean)

    if kept == 0:
        # Extremely defensive: if the remote set somehow cleaned down to nothing,
        # rebuild from the synthetic set which is guaranteed in-vocab and short.
        print("no examples survived cleaning; rebuilding from synthetic set.")
        tokens, targets, kept = build_examples(synthetic_pairs(), tok, clean)

    tokens_t = torch.tensor(tokens, dtype=torch.long)
    targets_t = torch.tensor(targets, dtype=torch.long)

    # Sanity: how much of the corpus is actually supervised (not -1)?
    supervised = int((targets_t != -1).sum().item())
    total = targets_t.numel()
    torch.save({"tokens": tokens_t, "targets": targets_t}, CHAT_PATH)

    print("-" * 60)
    print(f"examples kept        : {kept:,}")
    print(f"total chars (tokens) : {total:,}")
    print(f"supervised targets   : {supervised:,} ({100.0 * supervised / max(total, 1):.1f}% of positions)")
    print(f"masked targets (-1)  : {total - supervised:,}")
    print(f"wrote {CHAT_PATH}")
    print("next: python -m nanobdh.finetune --model gpt --max_iters 800")


if __name__ == "__main__":
    main()
