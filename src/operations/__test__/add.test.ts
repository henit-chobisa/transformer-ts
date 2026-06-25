import { describe, it, expect } from "vitest";
import { Value } from "../../grad";
import { addMatrix } from "../add";

const toValues = (m: number[][]): Value[][] => m.map((row) => row.map((n) => new Value(n)));
const nums = (m: Value[][]): number[][] => m.map((row) => row.map((v) => v.value));

describe("addMatrix", () => {
  it("adds two matrices element-wise", () => {
    const a = toValues([
      [1, 2],
      [3, 4],
    ]);
    const b = toValues([
      [10, 20],
      [30, 40],
    ]);

    expect(nums(addMatrix(a, b))).toEqual([
      [11, 22],
      [33, 44],
    ]);
  });

  it("preserves the shape (N x d -> N x d)", () => {
    const a = toValues([
      [1, 2, 3],
      [4, 5, 6],
    ]);
    const b = toValues([
      [1, 1, 1],
      [1, 1, 1],
    ]);

    const out = addMatrix(a, b);
    expect(out.length).toBe(2); // rows
    expect(out[0]!.length).toBe(3); // cols
  });

  it("handles negative numbers", () => {
    const a = toValues([
      [5, -3],
      [-1, 0],
    ]);
    const b = toValues([
      [-5, 3],
      [1, 0],
    ]);

    expect(nums(addMatrix(a, b))).toEqual([
      [0, 0],
      [0, 0],
    ]);
  });

  it("backprops gradient of 1 to BOTH inputs at each position", () => {
    // out[i][j] = a[i][j] + b[i][j]
    // d(out)/d(a[i][j]) = 1,  d(out)/d(b[i][j]) = 1
    const a = toValues([
      [2, 3],
      [4, 5],
    ]);
    const b = toValues([
      [1, 1],
      [1, 1],
    ]);

    const out = addMatrix(a, b);

    // sum all outputs into one scalar and backprop
    let loss = new Value(0);
    for (const row of out) for (const v of row) loss = loss.add(v);
    loss.backward();

    // every element of a and b contributed once → grad 1
    for (const row of a) for (const v of row) expect(v.grad).toBe(1);
    for (const row of b) for (const v of row) expect(v.grad).toBe(1);
  });

  it("is graph-connected (output traces back to inputs)", () => {
    const a = toValues([[1, 2]]);
    const b = toValues([[3, 4]]);
    const out = addMatrix(a, b);

    // backprop from one output element
    out[0]![0]!.backward();
    // a[0][0] and b[0][0] should each have grad 1, others 0
    expect(a[0]![0]!.grad).toBe(1);
    expect(b[0]![0]!.grad).toBe(1);
    expect(a[0]![1]!.grad).toBe(0);
  });

  it("does NOT mutate the input matrices", () => {
    // residual is used a lot — addMatrix must not corrupt its inputs
    const a = toValues([
      [1, 2],
      [3, 4],
    ]);
    const b = toValues([
      [5, 6],
      [7, 8],
    ]);

    addMatrix(a, b);

    // a and b should be untouched
    expect(nums(a)).toEqual([
      [1, 2],
      [3, 4],
    ]);
    expect(nums(b)).toEqual([
      [5, 6],
      [7, 8],
    ]);
  });

  it("throws on shape mismatch", () => {
    const a = toValues([
      [1, 2],
      [3, 4],
    ]); // 2x2
    const b = toValues([[1, 2]]); // 1x2

    expect(() => addMatrix(a, b)).toThrow();
  });
});
