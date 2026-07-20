# Glossary - every buzzword, plainly

Defined in the order you meet them. Grows as we go. Plain first; deep dive in the
linked chapter.

### Neural network
A big function made of many simple numeric operations, whose behavior is
controlled by numbers called **weights**. We don't program the rules; we *learn*
the weights from data.

### Weights / parameters
The adjustable numbers inside the model (often millions of them). "Training" =
finding good values for these. Used interchangeably with **parameters**.

### Tensor
Just a multi-dimensional array of numbers (a scalar is 0-D, a vector 1-D, a matrix
2-D, and so on). PyTorch's basic object. All model math is tensor math.

### PyTorch
The Python library we use. It gives us tensors, **autograd** (it automatically
computes how to nudge each weight), and GPU acceleration.

### MPS (Metal Performance Shaders)
Apple's GPU backend. On a Mac, PyTorch can run on the built-in GPU via MPS, much
faster than the CPU.

### Language model
A model that predicts the next piece of text given the text so far. That single
skill, repeated, produces fluent writing. See [Ch.0](00-overview.md).

### Token / tokenization
The "pieces" of text the model reads. We use **character-level** tokens: each
character (letter, space, punctuation) is one token. Tokenization = converting
text into a sequence of integer token ids. See [Ch.1](01-data-and-tokenization.md).

### Vector
An ordered list of numbers, e.g. `[0.2, -1.1, 0.7, ...]`. The model represents
each token as a vector so it can do math on "meaning".

### Embedding
A lookup table that maps each token id to a learned vector. Similar tokens end up
with similar vectors. See [Ch.2](02-embeddings.md).

### Positional encoding
Extra information added to each token's vector telling the model **where** the
token is in the sequence (attention alone has no sense of order). See Ch.2.

### Self-attention
The Transformer's core operation: each token compares itself to other tokens and
pulls in information from the relevant ones. See [Ch.3](03-self-attention.md).

### Softmax
A function that turns a list of raw scores into probabilities that are all
positive and sum to 1. Used for attention weights and for the final
next-character distribution.

### Loss
A single number measuring how wrong the model's predictions are. Lower is better.
We use **cross-entropy** loss. Training = making the loss go down. See Ch.6.

### Gradient descent / backpropagation
The learning algorithm. **Backpropagation** computes, for every weight, which
direction changes the loss; **gradient descent** nudges each weight a small step
in the loss-reducing direction. Repeat. See [Ch.6](06-training.md).

### Training vs inference
**Training:** adjust weights to reduce loss on data. **Inference:** freeze weights
and just run the model forward to get outputs (e.g. generate text).

### Self-supervised learning
Training where the labels come free from the data itself - here, the "correct
next character" is just the character that actually follows. No human labeling.

### Sampling / generation
Using the model's next-character probabilities to pick a character, then repeating
to produce text. Knobs like **temperature** and **top-k** control randomness. See
[Ch.7](07-sampling.md).

### BDH (Dragon Hatchling)
A brain-inspired alternative to the Transformer: a field of neurons with sparse,
positive activity where attention emerges from local interactions. See
[Ch.8](08-bdh-the-twist.md).


## More terms (harvested from the chapters)

- TinyShakespeare: a ~1 MB plain-text file of Shakespeare used as the training data.
- token: one unit of text fed to the model; here it is exactly one character.
- character-level: tokenizing one character at a time instead of by word or subword.
- tokenization: converting text into a sequence of integer ids.
- vocabulary (vocab): the sorted list of all unique characters; its count is V (~65).
- V (vocab size): the number of distinct tokens the model can read or emit (~65 here).
- stoi: the string-to-integer lookup table mapping each character to its id.
- itos: the integer-to-string lookup table mapping each id back to its character.
- encode: turn a text string into a list of integer ids.
- decode: turn a list of integer ids back into a text string.
- training set: the 90 percent of tokens the model is allowed to learn from.
- validation set: the held-out 10 percent used only to test the model honestly.
- overfitting: memorizing training data rather than learning general patterns, seen as a train-vs-val gap.
- B (batch size): how many T-length snippets are processed in parallel per step.
- T (context length / block size): the number of tokens in one snippet; the model's maximum look-back.
- epoch: one full pass over the dataset (nano-bdh trains by iteration count instead).
- BPE (byte-pair encoding): the subword tokenizer real GPT-2 uses, deliberately avoided here for simplicity.
- Embedding dimension (C): the number of values in each token's meaning vector; the fixed width every layer works with.
- Token embedding table: a learned lookup table with one row-vector per token id (V rows by C columns); nn.Embedding(V, C).
- Meaning vector: the C-length vector a token id maps to, whose values training shapes so similar tokens sit close.
- Lookup: fetching a table row by integer index rather than computing it; embedding is a lookup, not a multiply.
- Position: the slot index (0 to T-1) a token holds in the current context window.
- Positional embedding: a learned table (T rows by C columns) giving each slot a vector, so the order-blind attention can sense order.
- Elementwise addition: combining two equal-length vectors by summing matching entries; how token and position vectors merge into one.
- Broadcasting: PyTorch stretching the (T,C) position tensor across the batch axis to add it to a (B,T,C) tensor.
- Learned positional embedding: GPT-2/nanoGPT style trainable position table (capped at T rows), versus Vaswani 2017's fixed sinusoidal formula.
- Attention head: one complete query-key-value matching plus value-mixing operation that lets each position pull information from earlier positions.
- Query (Q): a position's learned 'what am I looking for' vector, matched against keys.
- Key (K): a position's learned 'how to find me / what I offer' label vector, compared against queries.
- Value (V): a position's learned 'what I hand over when selected' content vector, kept separate from its key.
- Dot product: multiply two vectors element by element and sum the results; here it scores how well a query matches a key.
- Attention weights: softmax-normalized match scores, all positive and summing to 1, giving each past position's contribution.
- Scaled dot-product attention: the standard formula that divides query-key scores by sqrt(head_size) before softmax to keep gradients healthy.
- Causal mask: setting every future-looking score to negative infinity so a position attends only to itself and earlier positions (no peeking ahead).
- head_size (hs): the width of the Q, K, and V vectors inside a single attention head.
- Multi-head attention: running several attention heads in parallel and concatenating their outputs (introduced fully in Chapter 4).
- Transformer block: one repeatable unit made of multi-head attention plus a feed-forward network, each wrapped in its own LayerNorm and residual connection; keeps shape (B, T, C) so blocks stack.
- Head size: the per-head width, equal to C divided by n_head.
- Residual (skip) connection: computing x = x + f(x) so a layer only learns the change while the original signal takes a free path, keeping gradients strong in deep networks.
- Vanishing gradient: the failure mode where the backward learning signal shrinks toward zero across many layers; residual connections prevent it.
- LayerNorm: rescales one token's C features to roughly mean 0 and spread 1 (plus learned scale and shift), stabilizing training; applied per position, never mixing positions or batch.
- Pre-norm: applying LayerNorm before each sub-layer (as in GPT-2 and our model) instead of after the residual add, keeping the residual highway clean for gradients.
- Feed-forward network (MLP): a position-wise little network that expands C to 4C, applies a nonlinearity, and projects back to C, letting each position process what attention gathered.
- Linear layer: a learnable matrix multiply plus bias (nn.Linear) that transforms a vector from one size to another.
- GELU: the smooth activation function used in GPT-2's feed-forward, a softened version of ReLU.
- Activation function: a simple nonlinear bend applied to each number that lets a network learn curved, non-linear relationships.
- Dropout: randomly zeroing a fraction of activations during training only, to reduce overfitting on small datasets like TinyShakespeare.
- n_layer (N): the number of Transformer blocks stacked in the model
- stacking: chaining blocks so each one's output feeds the next and refinements accumulate
- final LayerNorm (ln_f): the single normalization after the last block, before the head, that standardizes the residual stream for the head to read
- LM head: the final linear layer that maps each position's C-dim vector to V next-character scores
- logits: raw unnormalized per-character scores of shape (B, T, V); softmax converts them to probabilities
- forward pass: one run of data from input token ids all the way to output logits
- weight tying: reusing the token embedding matrix as the LM head weight, saving V*C parameters
- parameter: a single learned number (one weight or bias); their sum is the model size
- residual stream: the running vector x that each block adds a correction to instead of overwriting
- cross-entropy loss: the negative log of the probability the model assigned to the correct next character, averaged over all predicted positions; measures how surprised the model was by the truth
- negative log likelihood: another name for the per-example cross-entropy term, -log(prob of the true label)
- parameters (weights): the adjustable numbers inside the model that training modifies to reduce loss
- gradient: for each weight, a number giving the direction and steepness of change in the loss if that weight is nudged
- backpropagation: the algorithm that efficiently computes the gradient of the loss with respect to every weight
- gradient descent: updating weights by stepping in the direction opposite the gradient to lower the loss
- learning rate: the size of each gradient-descent step; the single most important training hyperparameter
- Adam: an optimizer that adapts the step size per weight using running averages of past gradients and squared gradients
- AdamW: Adam with decoupled weight decay; the standard optimizer for GPT-style training
- weight decay: a gentle pull of every weight toward zero each step, used to reduce overfitting
- batch size (B): the number of training windows processed together in one step
- block size / context length (T): the number of characters the model sees at once in a window
- step (iteration): one full cycle of forward pass, loss, backward pass, and optimizer update
- train loss: the loss measured on data the model is actively trained on
- validation loss: the loss measured on held-out data never used for training, used to detect real generalization
- underfitting: when both train and validation loss stay high because the model has not learned enough
- hyperparameters: settings chosen before training (learning rate, B, T, C, n_head, n_layer, weight decay, steps) that the optimizer does not learn
- warmup: an early phase where the learning rate ramps up from near zero to its peak to stabilize fresh weights
- cosine decay: a schedule that smoothly lowers the learning rate toward a small floor over the course of training
- autoregressive generation: producing text one token at a time, feeding each generated token back in as context for the next
- sampling: choosing the next token at random weighted by its probability, instead of always taking the most likely one
- greedy decoding: always selecting the highest-probability token (argmax); deterministic and repetition-prone
- temperature: a divisor applied to logits before softmax that flattens (>1, more creative) or sharpens (<1, safer) the distribution
- top-k sampling: restricting the choice to the k most probable tokens, zeroing the rest, before sampling
- multinomial sampling: drawing an index from a probability vector in proportion to those probabilities (the weighted-die step)
- context crop / block_size: trimming fed-in context to the trained maximum length so untrained positions are never used
- Neuron (BDH): one number in the model's large activation list; 0 means off, a positive value means firing.
- ReLU: the rule max(0, x); keeps positive numbers, turns negatives into 0, producing positive and sparse activity.
- Sparse activation: only a small fraction (a few percent) of neurons are non-zero at any moment; the rest are exactly 0.
- Positive activation: neuron values are never negative, only 0 or above.
- Neuron dimension N: the large per-head space BDH expands into, many times bigger than the embedding dim C.
- Low-rank feed-forward: expanding C up to N and back down using skinny factored encoder/decoder matrices instead of one giant dense matrix.
- Encoder / decoder (BDH): the shared low-rank matrices that lift representations from C to N and project them back from N to C.
- Linear attention: attention carried as a fixed-size running state updated per position, cost roughly linear in T, with no softmax over all pairs and no growing KV-cache.
- Synapse / synaptic memory: the connection strengths between neurons; BDH stores recent context here instead of in a growing token buffer.
- Emergent attention: attention that arises from local neuron-to-neuron interactions rather than a separate purpose-built attention module.
- Superposition: a dense network packing many meanings into single overlapping numbers, hard to interpret; BDH avoids this by using sparse positive neurons.
- Activation: the value a neuron takes while the model reads a given text position
- Synapse: the weight (connection strength) between two neurons, stored in the model's weight matrices
- Neuron graph: dots (neurons) joined by lines (synapses), read directly from the trained weights
- Scale-free: a graph with a few highly connected hub nodes and many low-degree nodes; degree follows a power law
- Power law: a many-small-few-huge distribution that shows up as a straight line on a log-log plot
- Modular (community structure): clusters of neurons that connect mostly among themselves
- Modularity Q: a numeric score for how cleanly a graph divides into communities
- Monosemantic: a neuron or synapse that responds to exactly one interpretable concept
- Polysemantic: the opposite of monosemantic; one unit mixed up in many unrelated concepts at once
- N (neuron dimension): BDH's large activation space, several times bigger than the embedding dim C
- Emergent structure: patterns that arise on their own from training rather than from an explicit hand-written rule
- Fair comparison: holding data, tokenizer, size, and training recipe fixed so any measured difference comes only from the model design
- Parameter matching: sizing two models to roughly the same parameter count so neither wins merely by being bigger
- Activation sparsity: the property that most neurons are silent at any moment and only a small fraction are active
- Dense activations: the opposite of sparse, where most internal values are nonzero and can be positive or negative
- Neuron dimension (N): BDH's large internal width where its sparse, positive neuron field lives
- Nats per character: the unit of our cross-entropy loss; pure guessing on a 65-char vocab is about ln(65) = 4.17
