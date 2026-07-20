# Chapter 11 - Post-Training: from base model to assistant

![Base model to served assistant](diagrams/11-post-training.svg)

> Reference code: `data/prepare_chat.py` (builds the masked chat dataset), `nanobdh/finetune.py` (the supervised fine-tuning loop with loss masking), `serve.py` (a standard-library HTTP server that loads both chat checkpoints), and `web/index.html` (the two-panel comparison UI). The base checkpoints we start from are `out/gpt.pt` and `out/bdh.pt` from Chapter 6; we produce `out/gpt-chat.pt` and `out/bdh-chat.pt`.

## HARD REALITY, up front (read this first)

Everything in this chapter is a **method demonstration, not a capable product**. Our base model is roughly 0.8 to 1.6 million parameters, character-level, trained on about 1 MB of Shakespeare. Post-training will teach it the *shape of a conversation* - it will learn to see `User:` and reliably start writing after `Assistant:`, then stop at the next turn. That is a real, observable behavior change and it is the whole point of the chapter.

What it will **not** do is answer correctly. The content will be broken, dreamlike English, because the model has never read the everyday facts a real assistant needs, and it only knows about 65 characters of Shakespeare-flavored text. So the demo you build proves the *mechanism* of turning a base model into an assistant (the chat format, the masked loss, the serving loop). It does not, and cannot, produce competence. We say this loudly so nobody mistakes a working pipeline for a working assistant.

## 1. The everyday picture (no jargon yet)

Imagine someone who has read an enormous library and can continue any sentence you start in a plausible style. That is talent, but it is not the same as being *helpful*. If you walk up and ask "what is the capital of France?", they might just keep writing more of your sentence, or drift into a monologue, because nobody ever told them that a question is a cue to *stop and answer*. They have raw language ability but no manners, no sense of turn-taking, no instinct that "when a human asks, I reply, then I hand the floor back".

**Post-training** is finishing school for that person. We do not re-teach them language from scratch. We show them a few thousand short worked examples of the pattern "human asks X, a good assistant replies Y", and we grade them **only on the reply part**, never on the question. After enough examples, the instinct sticks: they see the question, they know it is their turn, they produce a reply, and they stop. The knowledge in their head is unchanged. What changed is the *behavior* - they now play the assistant role.

That is the entire idea of this chapter. A base model has the language ability; post-training gives it the conversation manners.

## 2. From zero: every term as it appears

### Base model vs assistant

A **base model** (also called a *pretrained* model) is what Chapter 6 produced: a next-character predictor trained on raw text. Its one and only skill is "given the text so far, guess what comes next". It is astonishingly good at *continuation* and completely innocent of *conversation*. Ask it a question and it does the only thing it knows: it continues the string, which usually means writing more text that looks like your prompt, not an answer.

An **assistant** (or *chat model*, or *instruction-following model*) is a base model that has been taught a new habit: recognize a question, produce a reply, and stop. Crucially, an assistant is not a different or bigger network. It is the *same weights*, gently adjusted so the *behavior* is helpful. This is why the file we save, `out/gpt-chat.pt`, has the exact same architecture and shape as `out/gpt.pt`. Only the numbers inside shifted.

### Pretraining vs post-training

**Pretraining** is the expensive first phase: read a mountain of text, learn language. It is where almost all the model's knowledge comes from, and it is what Chapter 6 did with `nanobdh/train.py`.

**Post-training** is everything you do *after* pretraining to shape behavior. It is cheap by comparison (fewer examples, fewer steps, smaller learning rate) because you are not teaching language, only nudging habits. Real labs spend enormous compute on pretraining and a comparatively tiny slice on post-training, yet the post-training is what makes the model feel like a helpful chatbot rather than an autocomplete engine. In our project, post-training is a single short fine-tuning run in `nanobdh/finetune.py`.

### SFT (Supervised Fine-Tuning), also called instruction tuning

**Fine-tuning** just means "keep training an already-trained model, on new data, usually with a small learning rate so you nudge rather than overwrite". **Supervised** means "we have labeled right answers to imitate" - here, human-written (or template-written) assistant replies. Put together, **SFT** = show the model a pile of `(question, good answer)` pairs and train it to reproduce the good answers.

When those pairs are phrased as instructions and responses ("Summarize this", "Translate that", "What is X?"), the same technique is often called **instruction tuning**. It is the *first* and most important post-training stage, and it is the only one we actually implement. SFT alone is enough to convert a base model into a rough assistant. This exact move - fine-tune a pretrained GPT on demonstration data - is Stage 1 of OpenAI's InstructGPT recipe (Ouyang et al., 2022), the paper that made "post-training" mainstream.

### Chat template and role markers

The model only sees a flat stream of characters. It has no built-in notion of "user" or "assistant". So we *invent* the roles in plain text and let the model learn to read them. A **chat template** is a fixed textual convention for laying out a conversation so that roles are visible in the character stream. **Role markers** are the literal strings that flag whose turn it is.

Our template, locked for this whole project, renders one example exactly like this (note the literal newlines and the blank line that ends the turn):

```
User: <question>
Assistant: <answer>

```

A multi-turn conversation is just these blocks stacked, and a *prompt for generation* leaves the last assistant turn open for the model to fill:

```
User: <q1>
Assistant: <a1>

User: <q2>
Assistant: 
```

The single most important design rule here (a **locked contract** in this codebase): we do **not** add special vocabulary tokens for the roles. Real tokenizers sometimes reserve dedicated tokens like `<|user|>`. We deliberately do not, because adding a token would resize the model's embedding table and make `out/gpt.pt` impossible to load. Instead, `User:` and `Assistant:` are ordinary characters that already exist in our ~65-character Shakespeare vocab (capital letters, lowercase, colon, space, newline are all in there). The role markers are just text the model learns to recognize. This is the whole reason the chat model can inherit the base model's weights unchanged.

### Loss masking on assistant turns (the key idea of the chapter)

This is the one concept that makes SFT *SFT* rather than ordinary training, so slow down here.

Recall from Chapter 6 that training loss is **cross-entropy**: at every position, `-log(probability the model gave to the true next character)`, averaged over all positions. If we fine-tuned on the full chat string naively, we would be grading the model on predicting *every* character - including the user's question and the literal `Assistant: ` prefix. But we do not want the model to learn to *write questions*. We want it to learn to write *answers*. Making it good at generating plausible user questions is at best wasted effort and at worst harmful (it might start answering itself).

The fix is **loss masking**: we compute the loss **only on the characters that belong to the assistant's reply**, and we tell the loss function to *ignore* every other position. "Ignore" has a precise implementation. For each example we build two equal-length tensors:

- a **token** tensor `x` = the full rendered chat string, as character ids (this is what the model reads),
- a **target** tensor `y` = the *next* character at each position for the parts we want graded, and the sentinel value **`-1`** at every position we want ignored.

The user turn and the literal `Assistant: ` prefix get target `-1`. The assistant's actual answer characters (including the blank-line terminator `\n\n` that ends the turn, so the model learns *where to stop*) get their real next-character targets. Then cross-entropy is told `ignore_index=-1`, which means "any position whose target is `-1` contributes exactly zero to the loss and zero to the gradient". The model is thus optimized purely to produce good assistant text, while still *reading* the question as context.

A tiny worked picture (`I` = ignored, target `-1`; `G` = graded):

```
text:    U  s  e  r  :     h  i     A  s  s  i  s  t  a  n  t  :     h  e  l  l  o \n \n
mask:    I  I  I  I  I  I  I  I  I  I  I  I  I  I  I  I  I  I  I  G  G  G  G  G  G  G
```

Only the `hello\n\n` region is graded. Everything up to and including `Assistant: ` is context the model conditions on but is never scored for.

One implementation subtlety specific to *our* codebase. Look at the model interface (Chapter 5): `model(idx, targets=None) -> (logits, loss)`, and its built-in `forward` computes plain cross-entropy with **no** `ignore_index`. So `nanobdh/finetune.py` cannot just pass targets and reuse the internal loss. Instead it calls `logits, _ = model(idx)` to get the `(B, T, V)` logits, then computes the masked loss itself:

```python
loss = F.cross_entropy(logits.view(-1, V), targets.view(-1), ignore_index=-1)
```

Same model, same forward, but the *caller* owns the masking. This keeps the base architecture untouched while adding the one behavior SFT needs.

### The dataset SFT needs

SFT does not need much data, but it needs the *right* data: short, clean examples of the exact behavior you want. For a helpful assistant that means simple everyday question-and-answer or instruction-and-response pairs, a few hundred to a few thousand of them. Quality and format-consistency matter far more than volume; a small set of well-formed demonstrations teaches the turn-taking habit reliably.

Two constraints are non-negotiable in our setup:

1. **Every character must be in the base vocab.** The model literally has no id for a character it never saw in Shakespeare, so `data/prepare_chat.py` cleans each example down to the ~65 allowed characters - dropping or replacing anything out-of-vocab, lowercasing where needed. An emoji, a curly quote, or a tab would break encoding.
2. **The script must always work, even offline.** `data/prepare_chat.py` first tries a small public Q/A set via a direct https download; if that fails, it falls back to a built-in synthetic set of a few hundred simple templated dialogues. Either way it renders each example with the template above, builds the masked `token`/`target` tensors, and saves `data/chat.pt` as `{"tokens": LongTensor, "targets": LongTensor}` (targets already carrying `-1` on the non-assistant positions). It prints how many examples and characters it produced so you can sanity-check.

## 3. Deeper dive

### Why masking, in gradient terms

A position with target `-1` produces zero loss, so its gradient contribution is zero, so it never nudges the weights. Practically, the model still *attends to* and *reads* those tokens (they are in `x`, they form the context that conditions later predictions), but the optimizer only ever hears "be better at the assistant characters". Over a fine-tuning run this concentrates the entire weight update on one behavior: emit a good reply after `Assistant:` and terminate at the turn boundary. If you forgot the mask and graded everything, a large fraction of your gradient budget would be spent teaching the model to *hallucinate user questions*, diluting the very habit you are trying to install.

### Where "stop" comes from

Nothing in the architecture knows about turns. The model learns to stop because we *graded it on stopping*: the assistant target region ends with the `\n\n` that closes the block, so the model is trained to emit that terminator after a complete reply. At generation time (in `serve.py`) we mirror this: we generate characters until the model produces the start of the next human turn (a newline followed by `User:`) or we hit `max_new_tokens`, then we strip any trailing `\nUser:` fragment so the returned reply is clean. Format learned during SFT and stop-condition enforced during serving are two halves of the same convention.

### Shapes, end to end, in B/T/C/V notation

Nothing about the tensor plumbing changes from Chapter 6; only the target tensor is different. Per batch inside `nanobdh/finetune.py`:

- `x`: `(B, T)` character ids of rendered chat examples (padded/packed to a common length `T`).
- `y`: `(B, T)` targets, identical in shape, but with `-1` on every non-assistant position.
- `model(x)` returns logits `(B, T, V)` (`V` about 65).
- Flatten to `(B*T, V)` logits and `(B*T,)` targets, call `F.cross_entropy(..., ignore_index=-1)`, get one scalar loss over the *graded* positions only, backprop, step. Same AdamW, same MPS-with-CPU-fallback device logic (`torch.backends.mps.is_available()`), just a smaller learning rate and fewer iterations than pretraining.

### Loading the base, saving the chat model

`nanobdh/finetune.py` **loads** `out/gpt.pt` / `out/bdh.pt`, reconstructs the exact architecture from the saved `config`, loads `state_dict`, fine-tunes with the masked loss, then **saves** `out/gpt-chat.pt` / `out/bdh-chat.pt` in the identical checkpoint format used by `nanobdh/train.py`:

```python
{"model": "gpt"|"bdh", "config": {...}, "vocab": [...chars...], "state_dict": ...}
```

Reusing the format means `serve.py` (and any sampler) can load a chat checkpoint with the same code path as a base checkpoint. No special casing, no resized embeddings, no new tokens - the locked contracts pay off here.

### The stages we do NOT implement: RLHF and DPO (an honest aside)

SFT is stage one of alignment. Real assistants usually go further, and it is worth knowing the map even though we stop at SFT.

- **RLHF (Reinforcement Learning from Human Feedback)** is the classic follow-on, and the heart of the InstructGPT recipe. It has two more stages after SFT: (2) collect human *rankings* of several model outputs and train a **reward model** to predict which reply people prefer; (3) use reinforcement learning (typically **PPO**, Proximal Policy Optimization) to push the SFT model toward replies the reward model scores highly. SFT teaches the model to *imitate* good answers; RLHF teaches it to *optimize for human preference*, catching things imitation misses (tone, safety, refusing bad requests). Ouyang et al. (2022) showed this three-stage pipeline made a 1.3B model that humans preferred over the raw 175B GPT-3 - alignment, not size, did the work.

- **DPO (Direct Preference Optimization)** is a newer, simpler alternative (Rafailov et al., 2023). It uses the same *kind* of preference data - pairs of "preferred" vs "rejected" replies - but skips the separate reward model and the RL loop entirely. Instead it folds preference optimization into a single closed-form classification-style loss that directly raises the probability of the preferred reply and lowers the rejected one, relative to a frozen reference model. It is more stable and cheaper to run than PPO-based RLHF, which is why many recent open models use it.

We implement **only SFT** because it is the one stage that is both conceptually clean *and* achievable on a laptop with a toy model, and because it alone is enough to demonstrate the base-to-assistant transformation. RLHF and DPO need preference data and machinery well beyond a teaching build, and at our scale they would have nothing meaningful to optimize.

### Serving and the dual UI

Once both chat checkpoints exist, `serve.py` (Python standard library only - `http.server` and `json`, no FastAPI/Flask/uvicorn) loads **both** `out/gpt-chat.pt` and `out/bdh-chat.pt` at startup and exposes:

- `GET /` serves `web/index.html` (and any split-out `app.js` / `style.css`),
- `POST /chat` with body `{"model": "gpt"|"bdh", "history": [{"role": ..., "content": ...}, ...]}` builds the prompt from `history` using the locked template, generates with the chosen model (temperature about 0.8, top_k about 40, up to about 200 new tokens), post-processes to a clean assistant reply, and returns `{"reply": str}`.

`web/index.html` is a plain two-panel chat: **left = "GPT (Transformer)"**, **right = "BDH (Dragon Hatchling)"**, one shared input box at the bottom. Sending a message appends it to both panels and fires `POST /chat` twice, once per model, so you watch the two architectures answer the *same* prompt side by side, each with a small stats line (params and base val loss: GPT 802K / 1.76, BDH 1.59M / 1.53) and a "thinking..." state while it waits. This turns the abstract Chapter 10 comparison into something you can poke at live.

And, to close the loop back to the top: what you will see in that UI is two little models that have genuinely learned to *take turns* and reply in the assistant slot, wrapped around content that is charming nonsense. That contrast - correct format, incompetent content - is exactly the lesson. Post-training installs behavior; it cannot conjure knowledge that pretraining never put there.

## 4. New terms recap

- **Base / pretrained model**: a raw next-character predictor (our `out/gpt.pt`, `out/bdh.pt`). Great at continuation, clueless about conversation.
- **Assistant / chat model**: the same weights, post-trained to take turns and reply (our `out/gpt-chat.pt`, `out/bdh-chat.pt`).
- **Pretraining vs post-training**: learn language (expensive, Chapter 6) vs shape behavior (cheap, this chapter).
- **Fine-tuning**: continue training an existing model on new data with a small learning rate.
- **SFT / instruction tuning**: supervised fine-tuning on `(question, good answer)` pairs; stage one of alignment and the only stage we implement.
- **Chat template**: a fixed plain-text layout that makes conversation roles visible in the character stream.
- **Role markers**: the literal `User:` and `Assistant:` strings; in-vocab plain text, never special tokens, so the embedding table is never resized.
- **Loss masking**: grading cross-entropy only on the assistant's reply, setting every other target to `-1` and using `ignore_index=-1`. The key idea of SFT.
- **Ignore index (`-1`)**: a sentinel target that makes a position contribute zero loss and zero gradient.
- **Turn terminator (`\n\n`)**: graded as part of the assistant reply so the model learns *where to stop*.
- **RLHF**: the two extra stages (reward model + PPO reinforcement learning) that optimize for human preference. Not implemented here.
- **Reward model**: a model trained on human rankings to score how good a reply is.
- **DPO**: a simpler, RL-free alternative to RLHF that optimizes preference pairs with one direct loss. Not implemented here.

---

**Sources:** [InstructGPT / RLHF three-stage pipeline overview](https://mbrenndoerfer.com/writing/rlhf-pipeline-sft-reward-model-ppo-training), [Ouyang et al. 2022 review](https://etcjournal.com/2025/07/29/a-review-of-ouyang-et-al-s-2022-paper-aka-instructgpt/), [Instruction tuning + RLHF explainer](https://medium.com/@akankshasinha247/instruction-tuning-rlhf-teaching-llms-to-follow-and-align-611a5462b1bf), [DPO vs RLHF comparative analysis](https://arxiv.org/pdf/2403.01857).

**Next:** the code chapters for `data/prepare_chat.py`, `nanobdh/finetune.py`, `serve.py`, and `web/index.html`, where these ideas become the actual files you run.