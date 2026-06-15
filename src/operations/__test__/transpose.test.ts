import { describe, it, expect } from "vitest";
import { Value } from "../../grad";
import { transpose } from "../transpose";

// small helper: pull the raw numbers out of a Value[][] so assertions are readable
const values = (m: Value[][]): number[][] => m.map((row) => row.map((v) => v.value));

describe("transpose", () => {
  it("flips a rectangular matrix (2x3 -> 3x2)", () => {
    // A is 2 rows x 3 cols
    const A: Value[][] = [
      [new Value(1), new Value(2), new Value(3)],
      [new Value(4), new Value(5), new Value(6)],
    ];

    const T = transpose(A);

    // result should be 3 rows x 2 cols
    expect(T.length).toBe(3); // rows
    expect(T[0]!.length).toBe(2); // cols

    // entry [i][j] of T should equal entry [j][i] of A
    expect(values(T)).toEqual([
      [1, 4],
      [2, 5],
      [3, 6],
    ]);
  });

  it("flips a square matrix (2x2)", () => {
    const A: Value[][] = [
      [new Value(1), new Value(2)],
      [new Value(3), new Value(4)],
    ];

    expect(values(transpose(A))).toEqual([
      [1, 3],
      [2, 4],
    ]);
  });

  it("transposing twice returns the original shape and values", () => {
    const A: Value[][] = [
      [new Value(7), new Value(8), new Value(9)],
      [new Value(1), new Value(2), new Value(3)],
    ];

    // (Aᵀ)ᵀ === A
    expect(values(transpose(transpose(A)))).toEqual(values(A));
  });

  it("preserves the SAME Value references (does not copy/clone)", () => {
    // critical for autograd: transpose must reuse the exact Value objects,
    // not create new ones — otherwise gradients won't flow back to the originals
    const a00 = new Value(1);
    const a01 = new Value(2);
    const a10 = new Value(3);
    const a11 = new Value(4);
    const A = [
      [a00, a01],
      [a10, a11],
    ];

    const T = transpose(A);

    // T[0][1] should be the EXACT same object as A[1][0]
    expect(T[0]![1]).toBe(a10); // toBe = reference equality
    expect(T[1]![0]).toBe(a01);
  });

  it("handles a single row (1xN -> Nx1)", () => {
    const A: Value[][] = [[new Value(1), new Value(2), new Value(3)]];

    const T = transpose(A);

    expect(T.length).toBe(3);
    expect(T[0]!.length).toBe(1);
    expect(values(T)).toEqual([[1], [2], [3]]);
  });
});
