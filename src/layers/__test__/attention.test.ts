import { describe, it, expect } from "vitest";
import { Value } from "../../grad";
import { Attention } from "../attention";

const toValues = (m: number[][]): Value[][] => m.map((row) => row.map((n) => new Value(n)));

describe("Attention (single head)", () => {
  it("output is N x outputDim when outputDim === inputDim", () => {
    const d = 4;
    const attn = new Attention(d, d); // square projection: d -> d
    const input = toValues([
      [1, 0, 1, 0],
      [0, 1, 0, 1],
      [1, 1, 0, 0],
    ]); // 3 words, 4 dims

    const out = attn.forward(input);

    expect(out.length).toBe(3); // N words preserved
    expect(out[0]!.length).toBe(4); // outputDim = 4
    expect(out[1]!.length).toBe(4);
    expect(out[2]!.length).toBe(4);
  });

  it("projects down to a smaller outputDim (d -> head_dim)", () => {
    // this is exactly how a head is used inside multi-head: d=4 in, head_dim=2 out
    const inputDim = 4;
    const outputDim = 2;
    const attn = new Attention(inputDim, outputDim);
    const input = toValues([
      [1, 2, 3, 4],
      [5, 6, 7, 8],
      [0, 1, 0, 1],
    ]); // 3 words, 4 dims

    const out = attn.forward(input);

    expect(out.length).toBe(3); // N words preserved
    expect(out[0]!.length).toBe(2); // projected down to outputDim = 2
    expect(out[1]!.length).toBe(2);
  });

  it("weight matrices have shape inputDim x outputDim", () => {
    const attn = new Attention(4, 2);
    // wq, wk, wv should each be 4 rows x 2 cols
    for (const w of [attn.wq, attn.wk, attn.wv]) {
      expect(w.length).toBe(4); // inputDim rows
      expect(w[0]!.length).toBe(2); // outputDim cols
    }
  });

  it("works for a different sequence length and dims", () => {
    const attn = new Attention(2, 2);
    const input = toValues([
      [1, 2],
      [3, 4],
    ]);
    const out = attn.forward(input);
    expect(out.length).toBe(2);
    expect(out[0]!.length).toBe(2);
  });

  it("produces finite numbers", () => {
    const attn = new Attention(3, 3);
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

  it("backprops gradients into wq, wk, wv", () => {
    const attn = new Attention(3, 3);
    const input = toValues([
      [1, 0, 2],
      [0, 1, 1],
    ]);

    const out = attn.forward(input);

    let loss = new Value(0);
    for (const row of out) for (const v of row) loss = loss.add(v);
    loss.backward();

    const hasGrad = (m: Value[][]) => m.some((r) => r.some((v) => v.grad !== 0));
    expect(hasGrad(attn.wq)).toBe(true);
    expect(hasGrad(attn.wk)).toBe(true);
    expect(hasGrad(attn.wv)).toBe(true);
  });

  it("gradients reach the input embeddings", () => {
    const attn = new Attention(2, 2);
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
