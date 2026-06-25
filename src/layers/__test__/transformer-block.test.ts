import { describe, it, expect } from "vitest";
import { Value } from "../../grad";
import { TransformerBlock } from "../transformer";
import { MLP } from "../mlp";

const toValues = (m: number[][]): Value[][] => m.map((row) => row.map((n) => new Value(n)));

describe("TransformerBlock", () => {
  it("output has the same shape as the input (N x d)", () => {
    const d = 4;
    const heads = 2;
    const block = new TransformerBlock(d, heads);

    const input = toValues([
      [1, 0, 1, 0],
      [0, 1, 0, 1],
      [1, 1, 0, 0],
    ]); // 3 words, 4 dims

    const out = block.forward(input);

    expect(out.length).toBe(3); // N preserved
    expect(out[0]!.length).toBe(4); // d preserved
    expect(out[1]!.length).toBe(4);
    expect(out[2]!.length).toBe(4);
  });

  it("works for a different sequence length and dimension", () => {
    const d = 6;
    const block = new TransformerBlock(d, 3); // head_dim = 2
    const input = toValues([
      [1, 2, 3, 4, 5, 6],
      [6, 5, 4, 3, 2, 1],
    ]);

    const out = block.forward(input);
    expect(out.length).toBe(2);
    expect(out[0]!.length).toBe(6);
  });

  it("produces finite numbers (residual + layernorm keep it stable)", () => {
    const block = new TransformerBlock(4, 2);
    const input = toValues([
      [10, -10, 5, -5],
      [1, 2, 3, 4],
      [0, 0, 0, 0],
    ]);

    const out = block.forward(input);
    for (const row of out) {
      for (const v of row) {
        expect(Number.isFinite(v.value)).toBe(true);
      }
    }
  });

  it("each output row has mean ~0 (final layernorm did its job)", () => {
    // the block ends with layernorm, so every output row should be ~centered
    const block = new TransformerBlock(4, 2);
    const input = toValues([
      [1, 2, 3, 4],
      [5, 6, 7, 8],
    ]);

    const out = block.forward(input);
    for (const row of out) {
      const mean = row.reduce((a, v) => a + v.value, 0) / row.length;
      expect(mean).toBeCloseTo(0, 4);
    }
  });

  it("backprops gradients into the attention weights", () => {
    const block = new TransformerBlock(4, 2);
    const input = toValues([
      [1, 0, 2, 1],
      [0, 1, 1, 0],
    ]);

    const out = block.forward(input);

    let loss = new Value(0);
    for (const row of out) for (const v of row) loss = loss.add(v);
    loss.backward();

    const hasGrad = (m: Value[][]) => m.some((r) => r.some((v) => v.grad !== 0));

    // every head's weights should have gradients
    for (const head of block.attention.heads) {
      expect(hasGrad(head.wq)).toBe(true);
      expect(hasGrad(head.wk)).toBe(true);
      expect(hasGrad(head.wv)).toBe(true);
    }
    // Wo too
    expect(hasGrad(block.attention.wo)).toBe(true);
  });

  it("backprops gradients into the MLP parameters", () => {
    const block = new TransformerBlock(4, 2);
    const input = toValues([
      [1, 2, 3, 4],
      [5, 6, 7, 8],
    ]);

    const out = block.forward(input);
    let loss = new Value(0);
    for (const row of out) for (const v of row) loss = loss.add(v.mul(v));
    loss.backward();

    // MLP.parameters() returns all weights + biases as Value[]
    const mlpParams = block.mlp.parameters();
    const someNonzero = mlpParams.some((p) => p.grad !== 0);
    expect(someNonzero).toBe(true);
  });

  it("gradients reach the input embeddings", () => {
    const block = new TransformerBlock(4, 2);
    const input = toValues([
      [1, 2, 3, 4],
      [5, 6, 7, 8],
    ]);
    const out = block.forward(input);
    let loss = new Value(0);
    for (const row of out) for (const v of row) loss = loss.add(v.mul(v));
    loss.backward();

    const inputHasGrad = input.some((r) => r.some((v) => v.grad !== 0));
    expect(inputHasGrad).toBe(true);
  });

  it("MLP params get gradients in isolation", () => {
    const mlp = new MLP(4, [4, 4]);
    const input = [new Value(1), new Value(2), new Value(3), new Value(4)];
    const out = mlp.forward(input); // Value[]

    let loss = new Value(0);
    for (const v of out) loss = loss.add(v);
    loss.backward();

    const params = mlp.parameters();
    const someNonzero = params.some((p) => p.grad !== 0);
    expect(someNonzero).toBe(true);
  });
});
