import { describe, it, expect } from "vitest";
import { Value } from "../../grad";
import { concat } from "../concat";

const toValues = (m: number[][]): Value[][] => m.map((row) => row.map((n) => new Value(n)));
const nums = (m: Value[][]): number[][] => m.map((row) => row.map((v) => v.value));

describe("concat", () => {
  it("joins two matrices horizontally (N x p + N x q -> N x (p+q))", () => {
    // head 1 (2x2)        head 2 (2x2)
    const a = toValues([
      [1, 2],
      [3, 4],
    ]);
    const b = toValues([
      [5, 6],
      [7, 8],
    ]);

    const out = concat([a, b]);

    expect(out.length).toBe(2); // rows unchanged (N)
    expect(out[0]!.length).toBe(4); // widths summed (2 + 2)
    expect(nums(out)).toEqual([
      [1, 2, 5, 6],
      [3, 4, 7, 8],
    ]);
  });

  it("joins three matrices (mimics 3 heads)", () => {
    const h1 = toValues([[1], [2]]); // 2x1
    const h2 = toValues([[3], [4]]); // 2x1
    const h3 = toValues([[5], [6]]); // 2x1

    const out = concat([h1, h2, h3]);

    expect(out[0]!.length).toBe(3); // 1 + 1 + 1
    expect(nums(out)).toEqual([
      [1, 3, 5],
      [2, 4, 6],
    ]);
  });

  it("joins matrices of different widths (N x 2 + N x 3 -> N x 5)", () => {
    const a = toValues([
      [1, 2],
      [3, 4],
    ]);
    const b = toValues([
      [5, 6, 7],
      [8, 9, 10],
    ]);

    const out = concat([a, b]);

    expect(out[0]!.length).toBe(5); // 2 + 3
    expect(nums(out)).toEqual([
      [1, 2, 5, 6, 7],
      [3, 4, 8, 9, 10],
    ]);
  });

  it("preserves the SAME Value references (does not clone)", () => {
    // critical for autograd: concat must reuse the exact objects,
    // or gradients won't flow back to the heads' weights
    const a00 = new Value(1);
    const a01 = new Value(2);
    const b00 = new Value(3);
    const a = [[a00, a01]];
    const b = [[b00]];

    const out = concat([a, b]);

    expect(out[0]![0]).toBe(a00); // exact same object
    expect(out[0]![1]).toBe(a01);
    expect(out[0]![2]).toBe(b00);
  });

  it("gradients flow back through the concatenated values", () => {
    // build two matrices, concat, sum everything, backward,
    // every original Value should get a gradient
    const a = toValues([[1, 2]]);
    const b = toValues([[3, 4]]);

    const out = concat([a, b]); // 1x4

    let loss = new Value(0);
    for (const row of out) for (const v of row) loss = loss.add(v);
    loss.backward();

    // each original value contributed once to the sum → grad 1
    expect(a[0]![0]!.grad).toBe(1);
    expect(a[0]![1]!.grad).toBe(1);
    expect(b[0]![0]!.grad).toBe(1);
    expect(b[0]![1]!.grad).toBe(1);
  });

  it("throws if matrices have different row counts", () => {
    const a = toValues([
      [1, 2],
      [3, 4],
    ]); // 2 rows
    const b = toValues([[5, 6]]); // 1 row

    expect(() => concat([a, b])).toThrow();
  });
});
