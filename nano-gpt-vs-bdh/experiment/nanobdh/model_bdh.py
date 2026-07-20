"""BDH: the "Dragon Hatchling", a brain-inspired language model.

This is a small, heavily-commented adaptation of the OFFICIAL reference
implementation at github.com/pathwaycom/bdh (Kosowski et al. 2025,
arXiv:2509.26507). The math below is faithful to that repo; the defaults are
shrunk so it trains on a Mac, and the comments are written for someone who has
never seen it before. Interfaces are kept parallel to model_gpt.py so the SAME
train.py / sample.py can drive either model.

BEGINNER FIRST
--------------
BDH plays the exact same game as GPT (predict the next character) but with a
different, biology-inspired mechanism. Three ideas make it tick:

  1. A HIGH-DIMENSIONAL NEURON SPACE.
     GPT works in a compact C-dim vector. BDH first BLOWS UP each token into a
     much larger space of N "neurons" (N is many times bigger than C), does its
     thinking there, then squeezes back down to C. Analogy: a small idea (C) is
     projected onto a huge wall of light bulbs (N); the pattern of which bulbs
     are ON is the computation.

  2. POSITIVE + SPARSE activations (ReLU).
     After projecting up, we apply ReLU, which clamps every negative value to 0.
     So neuron activations are non-negative and MOSTLY ZERO (sparse) - only a
     few bulbs light up at a time. This mirrors how real brains fire sparsely,
     and the paper argues it is what makes the model interpretable and modular.

  3. LINEAR ATTENTION as association.
     Instead of GPT's softmax attention, BDH uses "linear attention": each
     position accumulates a running sum of past information (no softmax, no
     T-by-T probability matrix in the classic sense). Think of it as memory that
     integrates the past, closer to how neurons pass signals along over time.

DEEPER DIVE (mapping to the reference code)
-------------------------------------------
Per layer, with encoder E (C->N per head), decoder Dec (N*nh -> C):

    x_sparse = relu(x @ E)                 # lift to neuron space, positive+sparse
    yKV      = LinearAttention(Q=x_sparse, K=x_sparse, V=x)   # causal, RoPE'd
    y_sparse = relu(LayerNorm(yKV) @ E_v)  # a second sparse neuron code
    xy       = x_sparse * y_sparse         # GATING: multiply the two sparse maps
    y        = LayerNorm(xy @ Dec)         # squeeze neuron space back down to C
    x        = LayerNorm(x + y)            # residual update (same idea as GPT)

Key differences from GPT you should notice:
  - There is ONE shared set of encoder/decoder matrices reused at every layer
    (the loop applies the same parameters n_layer times). GPT has independent
    weights per block. This weight-sharing is a defining BDH feature.
  - Attention uses RoPE (rotary position embeddings) baked into Q/K, and a
    strictly-lower-triangular mask (diagonal=-1): a position attends to STRICTLY
    earlier positions. No separate learned position table.
  - LayerNorm here has NO learnable scale/bias (elementwise_affine=False); it is
    a pure normalization, applied liberally to keep the big-N activations stable.
  - The multiplicative gate x_sparse * y_sparse is BDH's nonlinearity of choice,
    replacing GPT's GELU-MLP.

Notation: B=batch, T=context, C=n_embd, V=vocab, nh=n_head,
          N = neuron_dim_multiplier * C / nh  (neurons per head).
"""

from __future__ import annotations

import dataclasses
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclasses.dataclass
class BDHConfig:
    # Field names kept parallel to GPTConfig so train.py can treat them alike.
    vocab_size: int = 65    # V
    block_size: int = 128   # T - used only to crop context during generation
    n_layer: int = 4        # number of times the shared block is applied (train.py sets this)
    n_head: int = 4         # nh - parallel attention/neuron heads
    n_embd: int = 128       # C - the compact embedding width
    dropout: float = 0.1
    # How much bigger the neuron space N is than C (per head). This is the single
    # knob that controls BDH's "high dimensionality". The reference uses 128; we
    # default to 32 so it trains quickly on a Mac. N = mult * C / nh.
    neuron_dim_multiplier: int = 32


def _neuron_dim(config: BDHConfig) -> int:
    """N = neurons per head = neuron_dim_multiplier * C / nh (integer)."""
    return config.neuron_dim_multiplier * config.n_embd // config.n_head


def get_freqs(n: int, theta: float, dtype: torch.dtype) -> torch.Tensor:
    """Rotary-position frequencies, verbatim in spirit from the reference repo.

    RoPE encodes "where a token is" by ROTATING its Q/K vector by an angle that
    depends on the position. Different feature pairs rotate at different speeds
    (frequencies); this returns those per-feature frequencies. The `quantize`
    step pairs up adjacent features so each 2-D (cos/sin) pair shares a frequency.
    """
    def quantize(t, q=2):
        return (t / q).floor() * q

    return (
        1.0
        / (theta ** (quantize(torch.arange(0, n, 1, dtype=dtype)) / n))
        / (2 * math.pi)
    )


class LinearAttention(nn.Module):
    """BDH's causal, RoPE'd linear attention (adapted from the reference).

    "Linear" here means: no softmax over the scores. We form Q@K^T, keep only the
    strictly-lower-triangular part (each position sees STRICTLY earlier ones),
    and multiply by V. Because Q and K are the same non-negative sparse tensor,
    this behaves like an accumulate-the-past associative memory rather than a
    normalized probability mixture.
    """

    def __init__(self, config: BDHConfig):
        super().__init__()
        self.config = config
        N = _neuron_dim(config)
        # Precompute RoPE frequencies for the neuron dimension N. Buffer = moves
        # with .to(device) but is not trained.
        self.register_buffer(
            "freqs",
            get_freqs(N, theta=2 ** 16, dtype=torch.float32).view(1, 1, 1, N),
        )

    @staticmethod
    def _phases_cos_sin(phases: torch.Tensor):
        phases = (phases % 1) * (2 * math.pi)
        return torch.cos(phases), torch.sin(phases)

    @staticmethod
    def _rope(phases: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Apply the rotation: rotate each adjacent (even, odd) feature pair."""
        v_rot = torch.stack((-v[..., 1::2], v[..., ::2]), dim=-1).view(*v.size())
        cos, sin = LinearAttention._phases_cos_sin(phases)
        return (v * cos).to(v.dtype) + (v_rot * sin).to(v.dtype)

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        # In BDH, Q and K are literally the same sparse activation tensor.
        assert K is Q, "BDH uses the same sparse tensor for Q and K"
        _, _, T, _ = Q.size()

        # Position-dependent rotation angles for each of the T positions.
        r_phases = (
            torch.arange(0, T, device=self.freqs.device, dtype=self.freqs.dtype)
            .view(1, 1, -1, 1)
        ) * self.freqs
        QR = self._rope(r_phases, Q)  # rotated queries
        KR = QR                       # keys are identical (Q is K)

        # scores[i, j] = QR_i . KR_j, then mask to strictly-earlier j (tril -1).
        scores = (QR @ KR.mT).tril(diagonal=-1)  # (B, nh, T, T)
        return scores @ V             # gather past values -> (B, nh, T, C)


class BDH(nn.Module):
    """The full Dragon Hatchling model, parallel in interface to GPT."""

    def __init__(self, config: BDHConfig):
        super().__init__()
        assert config.vocab_size is not None
        self.config = config
        nh = config.n_head
        D = config.n_embd
        N = _neuron_dim(config)

        # SHARED parameters, reused at every layer (a hallmark of BDH):
        #   encoder   : lifts x (C) up into the neuron space (N) per head.
        #   encoder_v : a second lift used on the attention output.
        #   decoder   : squeezes the gated neuron code (N*nh) back down to C.
        self.encoder = nn.Parameter(torch.zeros((nh, D, N)).normal_(std=0.02))
        self.encoder_v = nn.Parameter(torch.zeros((nh, D, N)).normal_(std=0.02))
        self.decoder = nn.Parameter(torch.zeros((nh * N, D)).normal_(std=0.02))

        self.attn = LinearAttention(config)

        # LayerNorm with NO learnable affine params: pure normalization, used a lot
        # to keep the large neuron-space activations numerically stable.
        self.ln = nn.LayerNorm(D, elementwise_affine=False, bias=False)
        self.embed = nn.Embedding(config.vocab_size, D)
        self.drop = nn.Dropout(config.dropout)

        # Output projection C -> V (the LM head), as a raw parameter matrix.
        self.lm_head = nn.Parameter(torch.zeros((D, config.vocab_size)).normal_(std=0.02))

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        """idx: (B, T) token ids. Returns (logits, loss), same contract as GPT."""
        C = self.config
        B, T = idx.size()
        D = C.n_embd
        nh = C.n_head
        N = _neuron_dim(C)

        # Embed and add a head axis: (B, T, D) -> (B, 1, D) broadcast over heads.
        x = self.embed(idx).unsqueeze(1)  # (B, 1, T, D)
        x = self.ln(x)                    # normalize before the stack (helps a lot)

        # The SAME block applied n_layer times (shared weights across depth).
        for _ in range(C.n_layer):
            # 1. Lift x into the high-dim neuron space, then ReLU -> positive+sparse.
            x_latent = x @ self.encoder       # (B, nh, T, N)
            x_sparse = F.relu(x_latent)       # non-negative, mostly zero

            # 2. Linear attention: mix in information from strictly-earlier tokens.
            #    Q=K=x_sparse (association), V=x (what to carry forward).
            yKV = self.attn(Q=x_sparse, K=x_sparse, V=x)  # (B, nh, T, D)
            yKV = self.ln(yKV)

            # 3. A second sparse neuron code from the attention output.
            y_latent = yKV @ self.encoder_v   # (B, nh, T, N)
            y_sparse = F.relu(y_latent)

            # 4. GATE: multiply the two sparse maps elementwise. This is BDH's
            #    nonlinearity - a neuron contributes only if BOTH codes fire.
            xy_sparse = x_sparse * y_sparse   # (B, nh, T, N)
            xy_sparse = self.drop(xy_sparse)

            # 5. Squeeze the gated neuron space (all heads concatenated) back to C.
            yMLP = xy_sparse.transpose(1, 2).reshape(B, 1, T, N * nh) @ self.decoder
            y = self.ln(yMLP)                 # (B, 1, T, D)

            # 6. Residual update, normalized (same "annotate, do not overwrite" idea).
            x = self.ln(x + y)

        # Drop the head axis and project to per-character scores.
        logits = x.view(B, T, D) @ self.lm_head  # (B, T, V)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        """Same autoregressive loop as GPT. We crop context to block_size to keep
        the T-by-T attention affordable on a laptop, even though BDH's linear
        attention could in principle run over an unbounded history.
        """
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
