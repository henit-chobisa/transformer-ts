import { describe, it, expect } from "vitest";
import { Value } from "../../grad";
import { Attention } from "../attention";

// helper: wrap a number[][] as Value[][]
const toValues = (m: number[][]): Value[][] => m.map((row) => row.map((n) => new Value(n)));

describe("Attention (single head)", () => {
  it("output has the same shape as the input (N x d)", () => {
    const d = 4;
    const attn = new Attention(d);
    // 3 words, each a 4-dim embedding
    const input = toValues([
      [1, 0, 1, 0],
      [0, 1, 0, 1],
      [1, 1, 0, 0],
    ]);

    const out = attn.forward(input);

    expect(out.length).toBe(3); // N words in → N words out
    expect(out[0]!.length).toBe(4); // d features preserved
    expect(out[1]!.length).toBe(4);
    expect(out[2]!.length).toBe(4);
  });

  it("works for a different sequence length and dimension", () => {
    const d = 2;
    const attn = new Attention(d);
    const input = toValues([
      [1, 2],
      [3, 4],
    ]); // 2 words, 2-dim

    const out = attn.forward(input);
    expect(out.length).toBe(2);
    expect(out[0]!.length).toBe(2);
  });

  it("produces finite numbers (softmax + matmul stay stable)", () => {
    const attn = new Attention(3);
    const input = toValues([
      [1, 2, 3],
      [-1, 0, 1],
      [2, 2, 2],
    ]);

    const out = attn.forward(input);
    for (const row of out) {
      for (const v of row) {
        expect(Number.isFinite(v.value)).toBe(true);
      }
    }
  });

  it("backprops gradients into the weight matrices (wq, wk, wv)", () => {
    // The whole point: attention must be trainable.
    // Sum all outputs into one scalar, backward, and check the weights got gradients.
    const d = 3;
    const attn = new Attention(d);
    const input = toValues([
      [1, 0, 2],
      [0, 1, 1],
    ]);

    const out = attn.forward(input);

    // reduce the output matrix to a single Value to backprop from
    let loss = new Value(0);
    for (const row of out) {
      for (const v of row) {
        loss = loss.add(v);
      }
    }
    loss.backward();

    // every weight matrix should have received gradients (at least one nonzero)
    const hasGrad = (m: Value[][]) => m.some((r) => r.some((v) => v.grad !== 0));
    expect(hasGrad(attn.wq)).toBe(true);
    expect(hasGrad(attn.wk)).toBe(true);
    expect(hasGrad(attn.wv)).toBe(true);
  });

  it("gradients reach the input embeddings too", () => {
    const attn = new Attention(2);
    const input = toValues([
      [1, 2],
      [3, 4],
    ]);
    const out = attn.forward(input);

    let loss = new Value(0);
    for (const row of out) for (const v of row) loss = loss.add(v);
    loss.backward();

    const inputHasGrad = input.some((r) => r.some((v) => v.grad !== 0));
    expect(inputHasGrad).toBe(true);
  });
});
