import { describe, it, expect } from "vitest";
import { Embedding } from "../embedding";
import { Value } from "../../grad";

describe("Embedding", () => {
  it("returns one vector per token id, with correct dimension", () => {
    const emb = new Embedding(5, 4); // vocab 5, d 4
    const out = emb.forward([2, 0, 1]); // 3 tokens

    expect(out.length).toBe(3); // N tokens
    expect(out[0]!.length).toBe(4); // d
  });

  it("returns the actual table rows (same reference, so gradients reach the table)", () => {
    const emb = new Embedding(5, 4);
    const out = emb.forward([2]);
    expect(out[0]).toBe(emb.vocabularyVectors[2]); // exact same object
  });

  it("the same token id returns the same row each time", () => {
    const emb = new Embedding(5, 4);
    const out = emb.forward([3, 3]);
    expect(out[0]).toBe(out[1]); // both are table[3]
  });

  it("looks up the correct rows in order", () => {
    const emb = new Embedding(5, 4);
    const out = emb.forward([4, 1, 0]);
    expect(out[0]).toBe(emb.vocabularyVectors[4]);
    expect(out[1]).toBe(emb.vocabularyVectors[1]);
    expect(out[2]).toBe(emb.vocabularyVectors[0]);
  });

  it("gradients flow back into the embedding table", () => {
    const emb = new Embedding(5, 4);
    const out = emb.forward([2, 0]);

    let loss = new Value(0);
    for (const row of out) for (const v of row) loss = loss.add(v.mul(v)); // sum of squares
    loss.backward();

    // rows 2 and 0 were used → they should have gradients
    const rowHasGrad = (row: Value[]) => row.some((v) => v.grad !== 0);
    expect(rowHasGrad(emb.vocabularyVectors[2]!)).toBe(true);
    expect(rowHasGrad(emb.vocabularyVectors[0]!)).toBe(true);
  });

  it("unused rows get no gradient", () => {
    const emb = new Embedding(5, 4);
    const out = emb.forward([2]); // only row 2 used

    let loss = new Value(0);
    for (const row of out) for (const v of row) loss = loss.add(v.mul(v));
    loss.backward();

    // row 4 was NOT used → should stay zero
    const row4HasGrad = emb.vocabularyVectors[4]!.some((v) => v.grad !== 0);
    expect(row4HasGrad).toBe(false);
  });

  it("throws on out-of-bounds token id", () => {
    const emb = new Embedding(5, 4);
    expect(() => emb.forward([5])).toThrow(); // index 5 invalid for vocab 5 (valid: 0-4)
    expect(() => emb.forward([2, 7])).toThrow();
  });
});
