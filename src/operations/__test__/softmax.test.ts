import { describe, it, expect } from "vitest";
import { Value } from "../../grad";
import { softmax } from "../softmax";

const nums = (vs: Value[]): number[] => vs.map((v) => v.value);
const sum = (xs: number[]): number => xs.reduce((a, b) => a + b, 0);

describe("softmax", () => {
  it("outputs the same number of elements as the input", () => {
    const out = softmax([new Value(1), new Value(2), new Value(3)]);
    expect(out.length).toBe(3);
  });

  it("outputs values that sum to 1", () => {
    const out = softmax([new Value(1), new Value(2), new Value(3)]);
    expect(sum(nums(out))).toBeCloseTo(1);
  });

  it("outputs all positive values", () => {
    const out = softmax([new Value(-5), new Value(0), new Value(5)]);
    for (const v of out) {
      expect(v.value).toBeGreaterThan(0);
    }
  });

  it("equal inputs produce equal (uniform) probabilities", () => {
    // three equal scores → each should be 1/3
    const out = softmax([new Value(2), new Value(2), new Value(2)]);
    expect(out[0]!.value).toBeCloseTo(1 / 3);
    expect(out[1]!.value).toBeCloseTo(1 / 3);
    expect(out[2]!.value).toBeCloseTo(1 / 3);
  });

  it("larger input gets larger probability (order preserved)", () => {
    const out = softmax([new Value(1), new Value(3), new Value(2)]);
    // index 1 had the largest score → should have the largest probability
    expect(out[1]!.value).toBeGreaterThan(out[0]!.value);
    expect(out[1]!.value).toBeGreaterThan(out[2]!.value);
    // index 2 (score 2) should beat index 0 (score 1)
    expect(out[2]!.value).toBeGreaterThan(out[0]!.value);
  });

  it("matches hand-computed values for [1, 2, 3]", () => {
    // e^1=2.71828, e^2=7.38906, e^3=20.0855 ; sum=30.1928
    // probs: 0.09003, 0.24473, 0.66524
    const out = softmax([new Value(1), new Value(2), new Value(3)]);
    expect(out[0]!.value).toBeCloseTo(0.09003, 4);
    expect(out[1]!.value).toBeCloseTo(0.24473, 4);
    expect(out[2]!.value).toBeCloseTo(0.66524, 4);
  });

  it("gradients flow back to every input", () => {
    // sum a couple of outputs and backprop — every input should get a nonzero grad
    const inputs = [new Value(0.5), new Value(1.5), new Value(-0.5)];
    const out = softmax(inputs);

    // create a scalar to backprop from: out[0] + out[2]
    const scalar = out[0]!.add(out[2]!);
    scalar.backward();

    for (const inp of inputs) {
      expect(inp.grad).not.toBe(0); // gradient reached every input
    }
  });

  it("is graph-connected (output traces back to inputs as parents indirectly)", () => {
    // a softmax output, when backpropped, must change input grads —
    // here we just assert a single output backward produces finite grads
    const inputs = [new Value(1), new Value(2)];
    const out = softmax(inputs);
    out[0]!.backward();
    expect(Number.isFinite(inputs[0]!.grad)).toBe(true);
    expect(Number.isFinite(inputs[1]!.grad)).toBe(true);
  });
});
