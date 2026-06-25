import { describe, it, expect } from "vitest";
import { PositionalEncoding } from "../encoding.js";
import { Value } from "../../grad/value.js";

describe("PositionalEncoding", () => {
  it("returns one position vector per position, correct dimension", () => {
    const pos = new PositionalEncoding(10, 4); // maxSeqLen 10, d 4
    const out = pos.forward(3); // sequence of length 3

    expect(out.length).toBe(3); // 3 positions
    expect(out[0]!.length).toBe(4); // d
  });

  it("returns the FIRST `seqLength` rows of the table", () => {
    const pos = new PositionalEncoding(10, 4);
    const out = pos.forward(3);

    // should be exactly rows 0, 1, 2 of the position table (same references)
    expect(out[0]).toBe(pos.positionTable[0]);
    expect(out[1]).toBe(pos.positionTable[1]);
    expect(out[2]).toBe(pos.positionTable[2]);
  });

  it("returns the actual table rows (references preserved, so gradients reach the table)", () => {
    const pos = new PositionalEncoding(10, 4);
    const out = pos.forward(2);
    expect(out[0]).toBe(pos.positionTable[0]); // exact same object, not a copy
  });

  it("different sequence lengths return different counts", () => {
    const pos = new PositionalEncoding(10, 4);
    expect(pos.forward(1).length).toBe(1);
    expect(pos.forward(5).length).toBe(5);
    expect(pos.forward(10).length).toBe(10); // exactly maxSeqLen is allowed
  });

  it("gradients flow back into the position table", () => {
    const pos = new PositionalEncoding(10, 4);
    const out = pos.forward(2);

    let loss = new Value(0);
    for (const row of out) for (const v of row) loss = loss.add(v.mul(v)); // sum of squares
    loss.backward();

    const rowHasGrad = (row: Value[]) => row.some((v) => v.grad !== 0);
    expect(rowHasGrad(pos.positionTable[0]!)).toBe(true);
    expect(rowHasGrad(pos.positionTable[1]!)).toBe(true);
  });

  it("unused positions get no gradient", () => {
    const pos = new PositionalEncoding(10, 4);
    const out = pos.forward(2); // only positions 0,1 used

    let loss = new Value(0);
    for (const row of out) for (const v of row) loss = loss.add(v.mul(v));
    loss.backward();

    // position 5 not used → stays zero
    const row5HasGrad = pos.positionTable[5]!.some((v) => v.grad !== 0);
    expect(row5HasGrad).toBe(false);
  });

  it("throws when sequence length exceeds maxSeqLen", () => {
    const pos = new PositionalEncoding(10, 4);
    expect(() => pos.forward(11)).toThrow(); // 11 > maxSeqLen 10
  });
});
