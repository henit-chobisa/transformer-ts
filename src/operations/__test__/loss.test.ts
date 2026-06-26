import { describe, it, expect } from "vitest";
import { Value } from "../../grad";
import { crossEntropyLoss } from "../loss";

const toValues = (m: number[][]): Value[][] => m.map((row) => row.map((n) => new Value(n)));

describe("crossEntropyLoss", () => {
  it("returns a single scalar Value", () => {
    const loss = crossEntropyLoss(toValues([[1, 2, 3]]), [2]);
    expect(loss).toBeInstanceOf(Value);
  });

  it("near zero when confident AND correct", () => {
    // huge score on token 0, and token 0 IS the answer → -log(~1) ~ 0
    const loss = crossEntropyLoss(toValues([[100, 0, 0]]), [0]);
    expect(loss.value).toBeCloseTo(0, 2);
  });

  it("large when confidently WRONG", () => {
    // model sure it's token 0, but answer is token 2
    const loss = crossEntropyLoss(toValues([[100, 0, 0]]), [2]);
    expect(loss.value).toBeGreaterThan(5);
  });

  it("uniform logits give loss ~ ln(vocabSize)", () => {
    // equal scores → each prob 1/3 → -log(1/3) = ln(3) ≈ 1.0986
    const loss = crossEntropyLoss(toValues([[5, 5, 5]]), [1]);
    expect(loss.value).toBeCloseTo(Math.log(3), 3);
  });

  it("averages across multiple positions", () => {
    // both positions confident + correct → average ~0
    const loss = crossEntropyLoss(
      toValues([
        [100, 0, 0], // correct: 0
        [0, 100, 0], // correct: 1
      ]),
      [0, 1],
    );
    expect(loss.value).toBeCloseTo(0, 2);
  });

  it("backprops gradients to the logits", () => {
    const logits = toValues([[1, 2, 3]]);
    const loss = crossEntropyLoss(logits, [0]);
    loss.backward();
    expect(logits[0]!.some((v) => v.grad !== 0)).toBe(true);
  });

  it("correct token's logit gets a NEGATIVE gradient (loss wants to raise it)", () => {
    const logits = toValues([[1, 1, 1]]);
    const loss = crossEntropyLoss(logits, [0]); // token 0 correct
    loss.backward();
    expect(logits[0]![0]!.grad).toBeLessThan(0);
  });
});
