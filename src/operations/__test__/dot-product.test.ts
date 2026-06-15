import { describe, it, expect } from "vitest";
import { Value } from "../../grad";
import { dotProduct } from "../dot-product";

describe("dotProduct", () => {
  it("computes the correct scalar value", () => {
    // [2, 3, 4] · [1, 5, 2] = 2*1 + 3*5 + 4*2 = 2 + 15 + 8 = 25
    const a = [new Value(2), new Value(3), new Value(4)];
    const b = [new Value(1), new Value(5), new Value(2)];

    const result = dotProduct(a, b);

    expect(result.value).toBe(25);
  });

  it("handles a simple two-element case", () => {
    // [1, 2] · [3, 4] = 3 + 8 = 11
    const a = [new Value(1), new Value(2)];
    const b = [new Value(3), new Value(4)];

    expect(dotProduct(a, b).value).toBe(11);
  });

  it("backprops correct gradients to both inputs", () => {
    // result = a·b = a0*b0 + a1*b1
    // d(result)/d(a0) = b0,  d(result)/d(a1) = b1
    // d(result)/d(b0) = a0,  d(result)/d(b1) = a1
    const a = [new Value(2), new Value(3)];
    const b = [new Value(4), new Value(5)];

    const result = dotProduct(a, b);
    result.backward();

    // gradient of result wrt each a[i] should be b[i]
    expect(a[0]!.grad).toBe(4); // b0
    expect(a[1]!.grad).toBe(5); // b1
    // gradient of result wrt each b[i] should be a[i]
    expect(b[0]!.grad).toBe(2); // a0
    expect(b[1]!.grad).toBe(3); // a1
  });

  it("throws if the two arrays differ in length", () => {
    const a = [new Value(1), new Value(2)];
    const b = [new Value(1)];

    expect(() => dotProduct(a, b)).toThrow();
  });
});
