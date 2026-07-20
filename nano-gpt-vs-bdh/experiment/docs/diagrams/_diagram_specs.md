# Diagram specs per chapter (source for SVGs)

## Ch 01 - Data and Tokenization

One clean left-to-right pipeline figure titled "Chapter 1: text becomes B x T batches". Five stages as boxes connected by right-pointing arrows, each arrow labeled with the transforming action.

Box 1 (document icon), label: "TinyShakespeare\n~1 MB text\n'First Citizen: Before...'". Arrow labeled "download (data/prepare.py)".

Box 2, label: "Vocabulary\nsorted unique chars\nV ~= 65\nstoi / itos". Small callout underneath showing "'h' -> 46" and "46 -> 'h'" as a two-way mapping. Arrow labeled "encode() (tokenizer.py)".

Box 3, label: "One long token tensor\n1-D, ~1,000,000 int ids\n[18, 47, 56, 57, ...]". Arrow labeled "90 / 10 split by position".

Box 4, two stacked sub-boxes: top sub-box "Train set (first 90%)", bottom sub-box "Val set (last 10%) - never trained on". Arrow from the Train sub-box labeled "sample B random windows of length T".

Box 5, a grid drawn as B rows by T columns of little integer cells, label "Batch x: shape (B, T)". Directly below it a second identical grid shifted one cell to the right, label "Targets y: shape (B, T) = x shifted by 1". A curved annotation arrow from a cell x[0, t] to y[0, t] labeled "predict the NEXT char".

Bottom caption strip spanning the figure: "B = batch size, T = context length (look-back), V = vocab size. C (embedding dim) appears in Chapter 2." Use a light box outline style, monospace font for the integer cells, and color the Val sub-box in a muted/greyed tone to signal 'held out'.

## Ch 02 - Embeddings

A left-to-right flow diagram titled "Token + Position Embeddings" showing how one character becomes one summed vector.

Left column (input): a small vertical strip of 4 boxes representing a token-id sequence "h e l l" with ids "[46, 43, 50, 50]", labeled "idx, shape (B,T)". Below it a note "T = context length, example values shown for one sequence".

Two parallel lookup paths in the middle, each drawn as an arrow from the input into a table box:
- Upper path: arrow labeled "look up by token id" into a box labeled "Token embedding table  wte = nn.Embedding(V, C)  (V approx 65 rows)". Output arrow labeled "tok_emb, shape (B,T,C)" showing a row of C numbers, e.g. "[0.21, -1.10, ...]".
- Lower path: a separate small input strip "positions [0,1,2,3]" feeds an arrow labeled "look up by position" into a box labeled "Position embedding table  wpe = nn.Embedding(T, C)  (T rows)". Output arrow labeled "pos_emb, shape (T,C)".

Center-right: a circle with a plus sign "+" labeled "elementwise add (broadcast over B)". Both tok_emb and pos_emb arrows converge into it.

Right column (output): a box labeled "x = tok_emb + pos_emb, shape (B,T,C)" with an outgoing arrow labeled "into first Transformer block (Ch.3)".

Callouts: a small tag on the token table "meaning: what character"; a small tag on the position table "order: where in the window"; a footnote strip along the bottom "B=batch, T=context length, C=embedding dim, V=vocab size (approx 65). Both tables are learned. File: nanobdh/model_gpt.py".

Use two distinct accent colors for the two tables (e.g. blue for token, orange for position) and a neutral color for the summed output. Keep boxes rounded, arrows solid, single clean sans-serif font.

## Ch 03 - Self-Attention

Title: "One attention head, step by step (causal)". Layout: left-to-right pipeline in 6 stages, all operating on a short character sequence example so it feels concrete.

Stage 1 (far left): a vertical stack of 4 small boxes labeled with characters "s", "w", "o", "r" (positions T=1..4), collectively bracketed as "Input x, shape (B,T,C)".

Stage 2: three parallel arrows from the input stack into three side-by-side rounded boxes labeled "W_q (learned)", "W_k (learned)", "W_v (learned)". Their outputs are three labeled columns: "Q (B,T,hs)", "K (B,T,hs)", "V (B,T,hs)". Caption under this stage: "three learned linear projections".

Stage 3: a 4x4 grid (T by T) labeled "scores = Q Kᵀ / sqrt(hs), shape (B,T,T)". Draw arrows from Q and K columns feeding into the grid. Each grid cell is a dot-product match score.

Stage 4: same 4x4 grid but now the strictly-upper-right triangle cells are shaded/greyed and filled with the symbol "-inf", lower triangle plus diagonal left white. Label: "causal mask: future set to -inf (no peeking ahead)".

Stage 5: the grid after a "softmax (per row)" box, now labeled "attention weights, rows sum to 1"; show one highlighted row (e.g. row for "r") with a large weight arrow pointing back to an earlier position to illustrate weighted focus.

Stage 6 (far right): "weighted sum: weights @ V" box, output stack of 4 boxes labeled "out (B,T,hs) - each position now carries context from its past". 

Bottom banner spanning the whole figure: the formula "softmax( (Q Kᵀ)/sqrt(hs) + mask ) V". Use a small legend box: B=batch, T=context length, C=embed dim, hs=head size. Keep arrows left-to-right, monochrome with one accent color for the highlighted attention row and for the -inf masked cells.

## Ch 04 - The Transformer Block

One clean top-to-bottom figure titled "One Transformer Block" showing the data flowing down through a single block, with the residual highway drawn as a visible side rail. Layout, top to bottom:

- Top node: input box labeled "x  (B, T, C)".
- From x, a vertical MAIN arrow goes down into sub-layer 1. Simultaneously a thin curved SKIP arrow branches off x on the right side and bypasses the whole sub-layer (this is the residual highway; label it "residual / skip: keep original").
- Sub-layer 1 stack (draw as a vertical mini-column of small boxes): box "LayerNorm" -> box "Multi-Head Attention (n_head heads in parallel, each C/n_head wide)" -> small box "output linear (C->C)".
- A circled plus sign "(+)" where the main arrow out of sub-layer 1 meets the skip arrow. Output arrow from the (+) labeled "(B, T, C)".
- That sum becomes the input to sub-layer 2, and again a thin curved SKIP arrow branches right and bypasses it.
- Sub-layer 2 stack: box "LayerNorm" -> box "Feed-Forward / MLP: Linear C->4C, GELU, Linear 4C->C".
- A second circled plus sign "(+)" merging the sub-layer 2 output with its skip arrow.
- Bottom output box labeled "x  (B, T, C)  -> next block".

Visual notes: use one accent color for the two curved residual skip arrows to make the highway pop; put a small inset on the right of the Multi-Head Attention box showing 3 or 4 tiny parallel head rectangles labeled "head 1 ... head n_head" then a "concat" bar, to convey parallel heads. Annotate the equation form beside each (+): top one "x = x + Attn(LN(x))", bottom one "x = x + FFN(LN(x))". Keep everything monochrome except the residual rails. Emphasize that shape stays (B, T, C) throughout by repeating the (B, T, C) label at entry, both plus signs, and exit.

## Ch 05 - The Full GPT

One vertical top-to-bottom flow diagram titled "The Full GPT forward pass (shapes in B,T,C notation)".

Nodes, top to bottom, each drawn as a rounded box with the tensor shape labeled on the right edge of the arrow leaving it:
1. Box "Input token ids (idx)" -> arrow labeled "(B, T)"
2. Box "Token embedding wte  +  Position embedding wpe" -> arrow labeled "(B, T, C)"
3. A tall outer container box labeled "Stack of N = n_layer identical blocks" that encloses N stacked smaller boxes. Show 3 stacked block boxes with a vertical ellipsis between the 2nd and last to imply repetition. Each small block box internally shows two lines: "x = x + Attn(LN1(x))" and "x = x + MLP(LN2(x))". A thin straight line labeled "residual stream" runs down the left side through all blocks to show information flowing straight down. Arrow leaving the container labeled "(B, T, C)"
4. Box "Final LayerNorm (ln_f)" -> arrow labeled "(B, T, C)"
5. Box "Linear LM head (weight tied to wte)" -> arrow labeled "(B, T, V)"
6. Box "Logits" then a short arrow labeled "softmax" to a final box "Next-char probabilities (sum to 1)"

To the right, a small callout bracket spanning the whole diagram with text: "same skeleton reused by BDH; only block internals differ".

Bottom caption strip: "B=batch, T=context length, C=embed dim, V=vocab (~65). Char-level TinyShakespeare config: N=6, C=384, T=256, V=65, ~10.8M params."

Color hint: embeddings in a light blue, the block stack in a warm/highlighted color (it holds ~99% of parameters), norm and head in neutral gray. Left-side residual line in a contrasting accent so it reads as the conveyor belt.

## Ch 06 - Training

ONE horizontal training-loop figure, left to right, showing the cycle with a feedback arrow.

Boxes (left to right):
1. "train.pt (token tensor)" - a cylinder/data box on the far left.
2. "get_batch" box, with an arrow out labeled "x, y : (B, T)".
3. "Model (GPT or BDH)" box. Arrow out labeled "logits : (B, T, V)".
4. "Cross-entropy loss" box. Small callout beneath it: "-log(prob of correct char), averaged; step-0 approx 4.17". Arrow out labeled "loss (scalar)".
5. "Backprop" box. Arrow out labeled "gradients (one per weight)".
6. "AdamW step" box. Callout beneath: "weights -= lr * update ; + weight decay".

Feedback loop: a curved arrow from the "AdamW step" box back to the "Model" box, labeled "update weights, repeat". This closes the loop and is the visual focus.

Below the main loop, a separate small offshoot: a dashed arrow from the "Model" box down to a box labeled "estimate_loss (no_grad, eval)" which reads from two small cylinders "train.pt" and "val.pt" and outputs two labels "train loss" and "val loss" side by side, with a tiny note "gap growing = overfitting".

Color/style: main loop boxes in one accent color, the AdamW feedback arrow bold, the eval offshoot in a muted/dashed gray to signal it is a measurement side-path, not part of the learning update. Keep B/T/C/V labels on the arrows exactly as written.

## Ch 07 - Sampling

One clean left-to-right flow diagram titled "One step of the generation loop (shapes in B/T/C/V)". Boxes in sequence, each a rounded rectangle with a small shape annotation underneath:

1. Box "Context so far (token ids)" annotated "(B, T)". Arrow right, labeled "crop to last block_size".
2. Box "Model forward pass (model_gpt.py / model_bdh.py)" annotated "-> logits (B, T, V)". Arrow right labeled "take last time step logits[:, -1, :]".
3. Box "Logits for next char" annotated "(B, V), ~65 raw scores". Arrow right labeled "divide by temperature".
4. Box "Temperature scaling" annotated "T<1 peakier / T>1 flatter". Arrow right labeled "keep top k, set rest to -inf".
5. Box "Top-k crop" annotated "only k survivors". Arrow right labeled "softmax".
6. Box "Softmax -> probabilities" annotated "(B, V), sums to 1". Arrow right labeled "torch.multinomial".
7. Box "Sampled character id" annotated "(B, 1)".

Then a feedback arrow curving from box 7 back to box 1, labeled "append and repeat (T grows by 1)", to show the autoregressive loop. Below box 7 a small terminal box "Decode with CharTokenizer -> text (tokenizer.py)" reached by a dashed arrow labeled "after max_new_tokens".

Use a cool color for the model box (2), a warm/highlight color for the two knob boxes (4 Temperature and 5 Top-k) to emphasize they are the tunable controls, neutral gray for the rest. The looping feedback arrow should be visually prominent since it is the core idea of autoregression.

## Ch 08 - BDH: The Twist

One clean left-to-right flow diagram titled "One BDH layer vs one GPT layer" split into two horizontal lanes stacked vertically, sharing the same left input box and right output box so the contrast is obvious.

Shared left box: "Positions, width C" labeled "(B, T, C)".

TOP LANE (label the lane "GPT layer"): from the shared input, an arrow into a box "Self-attention: build Q, K, V (3 separate projections)"; from it a small side box "KV-cache (grows with T)" connected by a dashed arrow labeled "stores every past token"; then an arrow into "Softmax over all token pairs (cost T x T)"; then an arrow into "MLP: C -> 4C -> C, dense +/- activations"; arrow to a "+ residual, norm" merge node; arrow to the shared output box.

BOTTOM LANE (label the lane "BDH layer"): from the shared input, an arrow into box "Encoder: lift C -> N (low-rank)"; arrow into "ReLU -> positive + sparse neuron field (B, n_head, T, N)"; from there an arrow into "Linear attention: Q=K=sparse neurons, V=input; causal mask" with a small side box "Synaptic state (fixed size, runs over T)" connected by a dashed arrow labeled "memory in synapses"; then an arrow into "Element-wise gate x_sparse * y_sparse (stays sparse)"; then "Decoder: N -> C (low-rank, shared across layers)"; arrow to a "+ residual, norm" merge node; arrow to the shared output box.

Shared right output box: "Updated positions, width C" labeled "(B, T, C)", with a final downstream arrow to "LM head -> V scores (identical for both)".

Use color/shading to distinguish: GPT boxes one tint, BDH boxes another, shared boxes neutral. Annotate the two side boxes (KV-cache vs synaptic state) as the key contrast. Keep arrows single-direction, minimal crossings.

## Ch 09 - The Ahas

One figure, three stacked panels sharing a top-down reading order, titled "Three ahas of a trained BDH." Layout is top-down across the three panels; each panel reads left-to-right (structure on the left, measurement/plot on the right). A thin vertical arrow runs down the left margin connecting Panel A to B to C, annotated "sparsity + positivity enable the graph and the single-concept edges."

Panel A (top), labeled "Aha 1: sparse positive activations". Draw a grid representing one sentence's activation slice x of shape (T, N-sample): x-axis is T positions (label a few columns with characters t, o, space, b, e), y-axis is a sample of about 12 of the N neurons. Most cells are light/empty (silent); only about 1 in 20 cells per column is a filled dark dot (firing). Side note: "min activation = 0 (ReLU), active fraction approx 5%". To the right of the grid, a small histogram icon with mass concentrated near 0.05, labeled "per-token active fraction".

Panel B (middle), labeled "Aha 2: emergent scale-free, modular neuron graph". Left sub-figure: a network of about 15 nodes where 2 nodes are large hubs with many lines and the rest have 1 to 3 lines; color the nodes in three groups (three communities) with faint dashed circles around each cluster. Right sub-figure: a small log-log plot, x-axis "degree k", y-axis "count of neurons", scattered points fitted by a straight descending line labeled "power law (scale-free)". An arrow from a small NxN weight-matrix grid icon points into the graph, labeled "read from weights".

Panel C (bottom), labeled "Aha 3: monosemantic synapse". Show two neurons i and j joined by one bold highlighted edge labeled "synapse (i,j)". Below it, a line of TinyShakespeare text, for example "ROMEO. But soft", where only the uppercase letters (R,O,M,E,O and B,S) are highlighted, with caption "this edge fires only for: UPPERCASE". Add a contrast tag: a faint second edge labeled "random edge = polysemantic (fires everywhere)".

## Ch 10 - GPT vs BDH: Comparison

One clean figure, laid out top-to-bottom as a "fair-fight" comparison panel.

TOP: a single shared box labeled "Shared setup (held fixed)" containing three small pill labels: "TinyShakespeare data (data/prepare.py: train.pt / val.pt)", "Char tokenizer, V is about 65 (tokenizer.py)", "Same train recipe, matched params (train.py)". One arrow labeled "batch (B,T)" fans DOWN out of this box into TWO parallel columns.

LEFT COLUMN header "GPT (model_gpt.py)": stacked boxes top to bottom: "Token + position embedding (B,T,C)" -> "Self-attention: T x T softmax, quadratic" -> "MLP, DENSE signed activations" -> "LM head -> logits (B,T,V)". Tag this column with a small badge "activations: dense, +/-".

RIGHT COLUMN header "BDH (model_bdh.py)": stacked boxes: "Embed, C is small (B,T,C)" -> "Project up to neuron dim N (B,T,N)" -> "ReLU -> POSITIVE + SPARSE, ~few % active" -> "Linear attention: running state, linear cost" -> "read-out -> logits (B,T,V)". Tag this column with a small badge "activations: sparse, positive".

BOTTOM: both columns' "logits (B,T,V)" arrows converge into a single box labeled "Same scoreboard" containing four bullet chips: "train/val loss (nats/char)", "sample quality (sample.py)", "activation sparsity %", "cost + interpretability". 

Use one accent color for GPT column, a different accent for BDH column, neutral gray for the shared top/bottom boxes. Straight orthogonal arrows, generous whitespace, no drop shadows.
