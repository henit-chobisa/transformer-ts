import { describe, it, expect } from "vitest";
import { Value } from "../../grad";
import { layernorm } from "../layernorm";

const nums = (m: Value[][]): number[][] => m.map((row) => row.map((v) => v.value));
const toValues = (m: number[][]): Value[][] => m.map((row) => row.map((n) => new Value(n)));

const meanOf = (xs: number[]): number => xs.reduce((a, b) => a + b, 0) / xs.length;
const stdOf = (xs: number[]): number => {
  const m = meanOf(xs);
  const variance = meanOf(xs.map((x) => (x - m) ** 2));
  return Math.sqrt(variance);
};

describe("layernorm", () => {
  it("preserves the shape (N x d -> N x d)", () => {
    const input = toValues([
      [1, 2, 3, 4],
      [10, 20, 30, 40],
    ]);
    const out = layernorm(input);
    expect(out.length).toBe(2);
    expect(out[0]!.length).toBe(4);
    expect(out[1]!.length).toBe(4);
  });

  it("each row has mean ~0 after normalization", () => {
    const input = toValues([
      [1, 2, 3, 4],
      [100, 102, 98, 101],
      [-5, 0, 5, 10],
    ]);
    const out = nums(layernorm(input));

    for (const row of out) {
      expect(meanOf(row)).toBeCloseTo(0, 4); // centered at 0
    }
  });

  it("each row has std ~1 after normalization", () => {
    const input = toValues([
      [1, 2, 3, 4],
      [100, 102, 98, 101],
    ]);
    const out = nums(layernorm(input));

    for (const row of out) {
      // with epsilon tiny, std should be very close to 1
      expect(stdOf(row)).toBeCloseTo(1, 2);
    }
  });

  it("normalizes each row INDEPENDENTLY (different scales -> same result shape)", () => {
    // two rows with very different magnitudes should both end up ~mean0/std1
    const input = toValues([
      [1, 2, 3], // small
      [1000, 2000, 3000], // huge
    ]);
    const out = nums(layernorm(input));

    // both rows, despite wildly different inputs, normalize to the same pattern
    // (since [1,2,3] and [1000,2000,3000] have the same *shape*, just scaled)
    expect(out[0]![0]).toBeCloseTo(out[1]![0]!, 3);
    expect(out[0]![1]).toBeCloseTo(out[1]![1]!, 3);
    expect(out[0]![2]).toBeCloseTo(out[1]![2]!, 3);
  });

  it("preserves order within a row (largest stays largest)", () => {
    const out = nums(layernorm(toValues([[3, 1, 4, 1, 5]])));
    const row = out[0]!;
    // index 4 was the largest input (5) -> should be largest output
    const maxIdx = row.indexOf(Math.max(...row));
    expect(maxIdx).toBe(4);
  });

  it("handles a constant row without NaN (epsilon guard)", () => {
    // all values equal -> variance 0 -> would divide by zero without epsilon
    const input = toValues([[5, 5, 5, 5]]);
    const out = nums(layernorm(input));

    for (const v of out[0]!) {
      expect(Number.isFinite(v)).toBe(true); // no NaN/Infinity
      expect(v).toBeCloseTo(0, 4); // (5-5)/sqrt(0+eps) = 0
    }
  });

  it("backprops gradients to every input", () => {
    const input = toValues([
      [1, 2, 3],
      [4, 5, 6],
    ]);
    const out = layernorm(input);

    let loss = new Value(0);
    for (const row of out) for (const v of row) loss = loss.add(v);
    loss.backward();

    const inputHasGrad = input.every((r) => r.every((v) => Number.isFinite(v.grad)));
    expect(inputHasGrad).toBe(true);
    // at least some gradient should be nonzero
    const someNonzero = input.some((r) => r.some((v) => v.grad !== 0));
    expect(someNonzero).toBe(true);
  });
});
