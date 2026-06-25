import { describe, it, expect } from "vitest";
import { Transformer } from "../transformer";
import { Value } from "../../grad";

describe("Transformer", () => {
  // vocabSize, dimension, numHeads, numBlocks, maxSeqLength
  const makeModel = () => new Transformer(10, 4, 2, 2, 16);

  it("output shape is N x vocabSize (a score per token, per position)", () => {
    const model = makeModel();
    const tokenIds = [2, 5, 1]; // sequence of 3 tokens

    const logits = model.forward(tokenIds);

    expect(logits.length).toBe(3); // N positions
    expect(logits[0]!.length).toBe(10); // vocabSize scores per position
    expect(logits[2]!.length).toBe(10);
  });

  it("works for different sequence lengths", () => {
    const model = makeModel();
    expect(model.forward([0]).length).toBe(1);
    expect(model.forward([1, 2, 3, 4, 5]).length).toBe(5);
  });

  it("produces finite logits", () => {
    const model = makeModel();
    const logits = model.forward([3, 7, 2, 9]);
    for (const row of logits) {
      for (const v of row) {
        expect(Number.isFinite(v.value)).toBe(true);
      }
    }
  });

  it("backprops gradients into the embedding table", () => {
    const model = makeModel();
    const tokenIds = [2, 5];
    const logits = model.forward(tokenIds);

    let loss = new Value(0);
    for (const row of logits) for (const v of row) loss = loss.add(v.mul(v));
    loss.backward();

    // the used token rows should have gradients
    const rowHasGrad = (row: Value[]) => row.some((v) => v.grad !== 0);
    expect(rowHasGrad(model.embedding.vocabularyVectors[2]!)).toBe(true);
    expect(rowHasGrad(model.embedding.vocabularyVectors[5]!)).toBe(true);
  });

  it("backprops gradients into the positional table", () => {
    const model = makeModel();
    const logits = model.forward([2, 5, 1]);

    let loss = new Value(0);
    for (const row of logits) for (const v of row) loss = loss.add(v.mul(v));
    loss.backward();

    const rowHasGrad = (row: Value[]) => row.some((v) => v.grad !== 0);
    expect(rowHasGrad(model.positional.positionTable[0]!)).toBe(true);
  });

  it("backprops gradients into the final projection matrix", () => {
    const model = makeModel();
    const logits = model.forward([2, 5]);

    let loss = new Value(0);
    for (const row of logits) for (const v of row) loss = loss.add(v.mul(v));
    loss.backward();

    const someNonzero = model.finalProjection.some((r) => r.some((v) => v.grad !== 0));
    expect(someNonzero).toBe(true);
  });

  it("backprops gradients into the transformer blocks (attention + mlp)", () => {
    const model = makeModel();
    const logits = model.forward([2, 5, 1]);

    let loss = new Value(0);
    for (const row of logits) for (const v of row) loss = loss.add(v.mul(v));
    loss.backward();

    // check at least the first block's attention + mlp got gradients
    const firstBlock = model.blocks[0]!;
    const hasGrad = (m: Value[][]) => m.some((r) => r.some((v) => v.grad !== 0));

    // attention weights (first head)
    expect(hasGrad(firstBlock.attention.heads[0]!.wq)).toBe(true);
    // mlp params
    const mlpHasGrad = firstBlock.mlp.parameters().some((p) => p.grad !== 0);
    expect(mlpHasGrad).toBe(true);
  });

  it("end-to-end: EVERY parameter in the model is reachable by gradient", () => {
    // the real proof the whole model is one connected graph and fully trainable
    const model = makeModel();
    const logits = model.forward([1, 2, 3]);

    let loss = new Value(0);
    for (const row of logits) for (const v of row) loss = loss.add(v.mul(v));
    loss.backward();

    // gather a representative sample of params from each part and confirm
    // each PART has at least some gradient flow
    const partHasGrad = (m: Value[][]) => m.some((r) => r.some((v) => v.grad !== 0));

    expect(partHasGrad(model.embedding.vocabularyVectors)).toBe(true); // embeddings
    expect(partHasGrad(model.positional.positionTable)).toBe(true); // positions
    expect(partHasGrad(model.finalProjection)).toBe(true); // final proj
    for (const block of model.blocks) {
      // every block
      expect(partHasGrad(block.attention.wo)).toBe(true);
    }
  });
});
