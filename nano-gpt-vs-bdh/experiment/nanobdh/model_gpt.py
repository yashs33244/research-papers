"""GPT: a clean, small, nanoGPT-style Transformer for character-level modeling.

BEGINNER FIRST (read this before any code)
-------------------------------------------
A language model plays one game: "given the characters so far, predict the next
character." A GPT does it with a stack of identical processing layers ("blocks"),
each of which does two things:

  1. ATTENTION - every position looks back at earlier positions and pulls in the
     information it needs. Analogy: reading a sentence, when you hit the word
     "it" you glance back to find what "it" refers to. Attention is that glance,
     done in parallel for every word, learned from data.

  2. MLP (a small feed-forward network) - each position then "thinks" privately
     about what it gathered. Analogy: after looking around, you mull it over.

Wrapped around those are three supporting ideas:
  - EMBEDDINGS: turn each character-id into a vector of C numbers (its meaning),
    and add a POSITION embedding so the model knows character #1 from character #5.
  - RESIDUAL connections: each block ADDS its output to its input (x = x + block(x))
    instead of replacing it. Analogy: sticky notes on a document - you annotate,
    you do not rewrite. This is what lets us stack many layers without the signal
    getting destroyed.
  - LAYER NORM: re-scale the numbers before each sub-step so they stay in a sane
    range and training stays stable.

At the very end a "LM head" projects each position's vector back to V numbers,
one score ("logit") per possible next character. Softmax turns those into
probabilities. Training nudges the weights so the correct next character gets a
higher probability.

DEEPER DIVE
-----------
This is the decoder-only Transformer of Vaswani et al. 2017 ("Attention Is All
You Need"), in the GPT-2 arrangement popularized by Radford et al. 2019 and
Karpathy's nanoGPT. Key specifics implemented below:
  - CAUSAL self-attention: position t may attend to positions <= t only. We
    enforce this with a lower-triangular mask so the model can never "cheat" by
    seeing the future token it is supposed to predict.
  - MULTI-HEAD: we split the C-dim into n_head independent subspaces so the model
    can attend to several kinds of relationship at once, then concatenate.
  - PRE-NORM blocks: LayerNorm is applied on the way INTO attention/MLP (x + attn(ln(x))),
    which trains more stably than the original post-norm design.
  - Weight tying: the token embedding matrix and the final LM head share weights
    (a standard GPT-2 trick that saves parameters and tends to help).

Notation: B=batch, T=context length, C=embedding dim, V=vocab, nh=n_head.
"""

from __future__ import annotations

import dataclasses
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclasses.dataclass
class GPTConfig:
    # Small defaults chosen to train comfortably on a Mac (MPS) in minutes.
    vocab_size: int = 65   # V - set for real from the tokenizer in train.py
    block_size: int = 128  # T - max context length the model can look back on
    n_layer: int = 4       # number of Transformer blocks stacked in depth
    n_head: int = 4        # attention heads per block (C must divide by this)
    n_embd: int = 128      # C - width of every token vector
    dropout: float = 0.1   # regularization: randomly zero activations while training


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention: the "look back and gather" step."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0, "C must be divisible by n_head"
        self.n_head = config.n_head
        self.n_embd = config.n_embd

        # One big linear layer produces Query, Key and Value for ALL positions at
        # once (hence 3 * C outputs). Q/K/V is the classic attention vocabulary:
        #   Query = "what am I looking for?"   Key = "what do I offer?"
        #   Value = "what I actually pass on if you attend to me."
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # Projection applied after mixing the heads back together.
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # The causal mask: a lower-triangular matrix of 1s. Registered as a buffer
        # (moves with .to(device) but is not a trained parameter). It forbids any
        # position from attending to positions AFTER it.
        mask = torch.tril(torch.ones(config.block_size, config.block_size))
        self.register_buffer("mask", mask.view(1, 1, config.block_size, config.block_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        nh = self.n_head
        hs = C // nh  # head size: each head works in a C/nh-dim subspace

        # Compute Q, K, V, then split the last dim into the three pieces.
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)  # each: (B, T, C)

        # Reshape (B, T, C) -> (B, nh, T, hs) so each head attends independently.
        q = q.view(B, T, nh, hs).transpose(1, 2)
        k = k.view(B, T, nh, hs).transpose(1, 2)
        v = v.view(B, T, nh, hs).transpose(1, 2)

        # Attention scores: how much should each query attend to each key?
        # Divide by sqrt(head size) so the dot products do not blow up as hs grows.
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(hs))  # (B, nh, T, T)

        # CAUSAL MASK: set scores for future positions to -inf so softmax gives
        # them zero weight. This is the single line that makes the model honest.
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)  # turn scores into weights that sum to 1
        att = self.attn_dropout(att)

        # Weighted sum of Values = the information each position gathered.
        y = att @ v  # (B, nh, T, hs)

        # Re-assemble the heads back into a single (B, T, C) tensor.
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    """Position-wise feed-forward net: the private "think it over" step.

    Standard GPT-2 shape: expand C -> 4C, apply a nonlinearity (GELU), shrink 4C -> C.
    The 4x widening gives the layer room to compute richer features.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    """One Transformer block = attention sub-layer + MLP sub-layer, pre-norm.

    Note the "x = x + ...": these are the RESIDUAL connections. Each sub-layer
    proposes an EDIT to x rather than a replacement, which keeps gradients flowing
    cleanly through a deep stack.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))  # gather info from other positions
        x = x + self.mlp(self.ln_2(x))   # then think about it locally
        return x


class GPT(nn.Module):
    """The full model: embeddings -> N blocks -> final norm -> LM head."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),   # token embeddings
            wpe=nn.Embedding(config.block_size, config.n_embd),   # position embeddings
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embd),                     # final layer norm
        ))
        # LM head: project each C-vector to V logits (scores over next characters).
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # WEIGHT TYING: reuse the token-embedding matrix as the output projection.
        # Same-sized matrices, and tying them saves parameters and usually helps.
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        # Small Gaussian init is what GPT-2 uses; keeps early activations tame.
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self) -> int:
        """Parameter count (minus position table, following nanoGPT convention)."""
        n = sum(p.numel() for p in self.parameters())
        n -= self.transformer.wpe.weight.numel()
        return n

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        """idx: (B, T) token ids. Returns (logits, loss).

        If `targets` is given (shape (B, T)), we also compute the cross-entropy
        training loss. If not (generation time), loss is None.
        """
        B, T = idx.size()
        assert T <= self.config.block_size, (
            f"sequence length {T} exceeds block_size {self.config.block_size}"
        )
        device = idx.device
        pos = torch.arange(0, T, dtype=torch.long, device=device)  # 0,1,...,T-1

        tok_emb = self.transformer.wte(idx)   # (B, T, C) "what" each token is
        pos_emb = self.transformer.wpe(pos)   # (T, C)    "where" each token sits
        x = self.transformer.drop(tok_emb + pos_emb)  # inject content + position

        for block in self.transformer.h:      # the deep stack of blocks
            x = block(x)
        x = self.transformer.ln_f(x)

        logits = self.lm_head(x)               # (B, T, V) next-char scores

        loss = None
        if targets is not None:
            # Cross-entropy compares predicted distribution against the true next
            # character. Flatten (B, T) into one long list of predictions/targets.
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
            )
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        """Autoregressively extend `idx` by `max_new_tokens` characters.

        Loop: feed context -> get logits for the LAST position -> sample one new
        character -> append it -> repeat. Because context can only grow to
        block_size, we crop it to the most recent T tokens each step.
        """
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size:]  # keep last T tokens
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature      # focus on next-char scores
            if top_k is not None:
                # Keep only the top_k most likely characters; zero out the rest.
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)  # sample one char
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
