# transformer.ts

A transformer, built from scratch in TypeScript. No PyTorch, no NumPy, no ML
libraries — just plain numbers, gradients, and a lot of comments.

I'm building this to *understand* how transformers actually work. Not to use one,
but to make every piece by hand — from the autograd engine all the way up. If you
read the source top to bottom, the comments are written like notes to my future
self.

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
operations     dot product · transpose · matmul · softmax
  ↓            the matrix machinery attention is built from
attention      (next)
```

## Run it

```bash
pnpm install

# train the MLP on a toy dataset, watch the loss fall
pnpm tsx src/examples/train-mlp.ts

# run the tests
pnpm test
```

## What lives where

| Path                | What it is                                              |
| ------------------- | ------------------------------------------------------- |
| `src/grad/`         | `Value` — the autograd engine and its backward pass     |
| `src/network/`      | `Neuron`, `Layer`, `MLP`                                |
| `src/operations/`   | `dotProduct`, `transpose`, `matmul`, `softmax`          |
| `src/examples/`     | runnable demos (start with `train-mlp.ts`)              |
| `notes/`            | the handwritten notes I worked through along the way    |

## Notes

The `notes/` folder is the math worked out by hand — the derivatives, the chain
rule, the topological sort for backprop. The code is the clean version of those
scribbles.

Built for learning. Slow on purpose.
