import { describe, it, expect } from "vitest";
import { Value } from "../../grad";
import { MultiHeadAttention } from "../multi-head-attention";

const toValues = (m: number[][]): Value[][] => m.map((row) => row.map((n) => new Value(n)));

describe("MultiHeadAttention", () => {
  it("output has the same shape as the input (N x d)", () => {
    const d = 4;
    const heads = 2; // head_dim = 2
    const mha = new MultiHeadAttention(d, heads);

    const input = toValues([
      [1, 0, 1, 0],
      [0, 1, 0, 1],
      [1, 1, 0, 0],
    ]); // 3 words, 4 dims

    const out = mha.forward(input);

    expect(out.length).toBe(3); // N words preserved
    expect(out[0]!.length).toBe(4); // d features preserved (concat back to d, then Wo)
    expect(out[1]!.length).toBe(4);
    expect(out[2]!.length).toBe(4);
  });

  it("works with a single head (degenerate case)", () => {
    const d = 4;
    const mha = new MultiHeadAttention(d, 1); // head_dim = 4
    const input = toValues([
      [1, 2, 3, 4],
      [5, 6, 7, 8],
    ]);

    const out = mha.forward(input);
    expect(out.length).toBe(2);
    expect(out[0]!.length).toBe(4);
  });

  it("works with more heads (4 heads, d=8, head_dim=2)", () => {
    const d = 8;
    const heads = 4;
    const mha = new MultiHeadAttention(d, heads);

    const input = toValues([
      [1, 2, 3, 4, 5, 6, 7, 8],
      [8, 7, 6, 5, 4, 3, 2, 1],
    ]);

    const out = mha.forward(input);
    expect(out.length).toBe(2);
    expect(out[0]!.length).toBe(8); // concat of 4 heads × 2 = 8, then Wo → 8
  });

  it("produces finite numbers", () => {
    const mha = new MultiHeadAttention(4, 2);
    const input = toValues([
      [1, -2, 3, -4],
      [0, 1, 0, 1],
      [2, 2, 2, 2],
    ]);

    const out = mha.forward(input);
    for (const row of out) {
      for (const v of row) {
        expect(Number.isFinite(v.value)).toBe(true);
      }
    }
  });

  it("backprops gradients into every head's weights", () => {
    const d = 4;
    const heads = 2;
    const mha = new MultiHeadAttention(d, heads);
    const input = toValues([
      [1, 0, 2, 1],
      [0, 1, 1, 0],
    ]);

    const out = mha.forward(input);

    // reduce output to a single scalar and backprop
    let loss = new Value(0);
    for (const row of out) for (const v of row) loss = loss.add(v);
    loss.backward();

    const hasGrad = (m: Value[][]) => m.some((r) => r.some((v) => v.grad !== 0));

    // every head's Q, K, V matrices should have received gradients
    for (const head of mha.heads) {
      expect(hasGrad(head.wq)).toBe(true);
      expect(hasGrad(head.wk)).toBe(true);
      expect(hasGrad(head.wv)).toBe(true);
    }
  });

  it("backprops gradients into Wo (the output projection)", () => {
    const mha = new MultiHeadAttention(4, 2);
    const input = toValues([
      [1, 2, 3, 4],
      [5, 6, 7, 8],
    ]);

    const out = mha.forward(input);
    let loss = new Value(0);
    for (const row of out) for (const v of row) loss = loss.add(v);
    loss.backward();

    const woHasGrad = mha.wo.some((r) => r.some((v) => v.grad !== 0));
    expect(woHasGrad).toBe(true);
  });

  it("throws when dimension is not divisible by number of heads", () => {
    expect(() => new MultiHeadAttention(10, 3)).toThrow(); // 10 / 3 is not integer
  });
});
