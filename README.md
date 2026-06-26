# transformer.ts

A transformer, built from scratch in TypeScript. No PyTorch, no NumPy, no ML
libraries — just plain numbers, gradients, and a lot of comments.

I'm building this to *understand* how transformers actually work. Not to use one,
but to make every piece by hand — from the autograd engine all the way up to a
model that trains. If you read the source top to bottom, the comments are written
like notes to my future self.

And it works: trained on `"hello world"`, the loss falls from **5.13 → 0.23**.
The whole thing is differentiable end to end, and every gradient is computed by
code I wrote.

## The build, layer by layer

Each piece only depends on the ones above it. That's the whole curriculum.

```
Value          an autograd engine — a number that remembers where it came from
  ↓            so it can compute its own gradient (backprop on a graph)
Neuron         weights · inputs + bias, squashed through tanh
  ↓
Layer          a row of neurons
  ↓
MLP            layers stacked — a tiny neural net that actually trains
  ↓
operations     dot product · transpose · matmul · softmax · add · concat
  ↓            layernorm · cross-entropy — the matrix machinery attention needs
Attention      Q · Kᵀ → softmax → weighted V. one head, one pattern
  ↓
MultiHead      many heads in parallel, concatenated, projected back
  +
Embedding      ids → vectors, plus a learned positional table so order matters
  ↓
TransformerBlock   attention + residual + layernorm + MLP + residual + layernorm
  ↓
Transformer    embeddings → blocks → projection to vocab logits
  ↓
training       cross-entropy loss, backward, nudge every weight. repeat.
```

## Run it

```bash
pnpm install

# train the full transformer on "hello world", watch the loss fall
pnpm tsx src/examples/train-transformer.ts

# the warm-up: train the MLP on a toy dataset
pnpm tsx src/examples/train-mlp.ts

# run the tests
pnpm test
```

## What lives where

| Path                | What it is                                                         |
| ------------------- | ----------------------------------------------------------------- |
| `src/grad/`         | `Value` — the autograd engine and its backward pass               |
| `src/operations/`   | `dotProduct`, `transpose`, `matmul`, `softmax`, `concat`, `layernorm`, `crossEntropyLoss` |
| `src/layers/`       | `Neuron`, `Layer`, `MLP`, `Attention`, `MultiHeadAttention`, `Embedding`, `PositionalEncoding`, `TransformerBlock`, `Transformer` |
| `src/examples/`     | runnable demos (`train-transformer.ts`, `train-mlp.ts`)           |
| `notes/`            | the handwritten notes I worked through along the way              |

Every file has a matching test in its `__test__/` folder — the operations and
layers are each checked on their own before they get stacked together.

## How it trains

The example feeds it a string, one character at a time. Each input token's job
is to predict the *next* token, so `"hello world"` becomes `inputIds` (everything
but the last char) and `targetIds` (everything but the first). Then it's the
usual loop: forward to logits, cross-entropy against the targets, `backward()` to
fill in every gradient, and a step down the slope for each parameter. A hundred
steps later the model has memorised the string.

## Notes

The `notes/` folder is the math worked out by hand — the derivatives, the chain
rule, the topological sort for backprop. The code is the clean version of those
scribbles.

Built for learning. Slow on purpose.
