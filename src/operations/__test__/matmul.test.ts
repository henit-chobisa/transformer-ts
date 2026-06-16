import { describe, it, expect } from "vitest";
import { Value } from "../../grad";
import { matmul } from "../matmul";

const values = (m: Value[][]): number[][] => m.map((row) => row.map((v) => v.value));

describe("matmul", () => {
  it("multiplies two 2x2 matrices correctly", () => {
    // A = [1 2]    B = [5 6]
    //     [3 4]        [7 8]
    //
    // out[0][0] = 1*5 + 2*7 = 19
    // out[0][1] = 1*6 + 2*8 = 22
    // out[1][0] = 3*5 + 4*7 = 43
    // out[1][1] = 3*6 + 4*8 = 50
    const A = [
      [new Value(1), new Value(2)],
      [new Value(3), new Value(4)],
    ];
    const B = [
      [new Value(5), new Value(6)],
      [new Value(7), new Value(8)],
    ];

    expect(values(matmul(A, B))).toEqual([
      [19, 22],
      [43, 50],
    ]);
  });

  it("multiplies non-square matrices (2x3 · 3x2 -> 2x2)", () => {
    // A is 2x3, B is 3x2  ->  output is 2x2
    // A = [1 2 3]    B = [7  8 ]
    //     [4 5 6]        [9  10]
    //                    [11 12]
    //
    // out[0][0] = 1*7 + 2*9  + 3*11 = 7 + 18 + 33 = 58
    // out[0][1] = 1*8 + 2*10 + 3*12 = 8 + 20 + 36 = 64
    // out[1][0] = 4*7 + 5*9  + 6*11 = 28 + 45 + 66 = 139
    // out[1][1] = 4*8 + 5*10 + 6*12 = 32 + 50 + 72 = 154
    const A = [
      [new Value(1), new Value(2), new Value(3)],
      [new Value(4), new Value(5), new Value(6)],
    ];
    const B = [
      [new Value(7), new Value(8)],
      [new Value(9), new Value(10)],
      [new Value(11), new Value(12)],
    ];

    const out = matmul(A, B);

    expect(out.length).toBe(2); // rows = A's rows
    expect(out[0]!.length).toBe(2); // cols = B's cols
    expect(values(out)).toEqual([
      [58, 64],
      [139, 154],
    ]);
  });

  it("output shape is (A rows) x (B cols)", () => {
    // A is 3x2, B is 2x4  ->  output must be 3x4
    const A = [
      [new Value(1), new Value(2)],
      [new Value(3), new Value(4)],
      [new Value(5), new Value(6)],
    ];
    const B = [
      [new Value(1), new Value(2), new Value(3), new Value(4)],
      [new Value(5), new Value(6), new Value(7), new Value(8)],
    ];

    const out = matmul(A, B);

    expect(out.length).toBe(3); // A's rows
    expect(out[0]!.length).toBe(4); // B's cols
  });

  it("identity matrix leaves the other matrix unchanged", () => {
    // A · I = A
    const A = [
      [new Value(2), new Value(7)],
      [new Value(5), new Value(3)],
    ];
    const I = [
      [new Value(1), new Value(0)],
      [new Value(0), new Value(1)],
    ];

    expect(values(matmul(A, I))).toEqual([
      [2, 7],
      [5, 3],
    ]);
  });

  it("backprops gradients to the inputs", () => {
    // Single output element to keep the gradient math simple:
    // A = [a b]  (1x2),  B = [c]  (2x1)
    //                        [d]
    // out[0][0] = a*c + b*d
    // d(out)/da = c,  d(out)/db = d
    // d(out)/dc = a,  d(out)/dd = b
    const a = new Value(2);
    const b = new Value(3);
    const c = new Value(4);
    const d = new Value(5);

    const A = [[a, b]]; // 1x2
    const B = [[c], [d]]; // 2x1

    const out = matmul(A, B); // 1x1
    out[0]![0]!.backward();

    expect(a.grad).toBe(4); // c
    expect(b.grad).toBe(5); // d
    expect(c.grad).toBe(2); // a
    expect(d.grad).toBe(3); // b
  });

  it("throws when A's columns do not match B's rows", () => {
    // A is 2x3, B is 2x2  -> inner dims (3 vs 2) mismatch
    const A = [
      [new Value(1), new Value(2), new Value(3)],
      [new Value(4), new Value(5), new Value(6)],
    ];
    const B = [
      [new Value(1), new Value(2)],
      [new Value(3), new Value(4)],
    ];

    expect(() => matmul(A, B)).toThrow();
  });
});
