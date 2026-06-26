import { describe, it, expect } from "vitest";
import { Value } from "../value";

describe("Value", () => {
  describe("construction", () => {
    it("holds a value, with grad initialized to 0 and no parents", () => {
      const x = new Value(5);
      expect(x.value).toBe(5);
      expect(x.grad).toBe(0);
      expect(x.parents).toEqual([]);
    });
  });

  describe("add", () => {
    it("computes the sum", () => {
      expect(new Value(2).add(new Value(3)).value).toBe(5);
    });

    it("records both operands as parents", () => {
      const a = new Value(2);
      const b = new Value(3);
      const c = a.add(b);
      expect(c.parents).toContain(a);
      expect(c.parents).toContain(b);
    });

    it("backprops gradient of 1 to each operand", () => {
      // d(a+b)/da = 1, d(a+b)/db = 1
      const a = new Value(2);
      const b = new Value(3);
      const c = a.add(b);
      c.backward();
      expect(a.grad).toBe(1);
      expect(b.grad).toBe(1);
    });
  });

  describe("mul", () => {
    it("computes the product", () => {
      expect(new Value(4).mul(new Value(5)).value).toBe(20);
    });

    it("backprops the other operand as each gradient", () => {
      // d(a*b)/da = b, d(a*b)/db = a
      const a = new Value(4);
      const b = new Value(5);
      const c = a.mul(b);
      c.backward();
      expect(a.grad).toBe(5); // b
      expect(b.grad).toBe(4); // a
    });
  });

  describe("tanh", () => {
    it("computes tanh of the value", () => {
      expect(new Value(0).tanh().value).toBe(0); // tanh(0) = 0
      expect(new Value(1).tanh().value).toBeCloseTo(Math.tanh(1));
    });

    it("backprops gradient of 1 - tanh^2", () => {
      // d/dx tanh(x) = 1 - tanh^2(x)
      const x = new Value(0.5);
      const out = x.tanh();
      out.backward();
      const expected = 1 - Math.tanh(0.5) ** 2;
      expect(x.grad).toBeCloseTo(expected);
    });
  });

  describe("exp", () => {
    it("computes e^x", () => {
      expect(new Value(0).exp().value).toBe(1); // e^0 = 1
      expect(new Value(1).exp().value).toBeCloseTo(Math.E);
    });

    it("backprops gradient equal to e^x (the output itself)", () => {
      const x = new Value(2);
      const out = x.exp();
      out.backward();
      expect(x.grad).toBeCloseTo(out.value);
      expect(x.grad).toBeCloseTo(Math.exp(2));
    });
  });

  describe("Value.div", () => {
    it("computes the quotient", () => {
      expect(new Value(10).div(new Value(2)).value).toBe(5);
      expect(new Value(7).div(new Value(2)).value).toBe(3.5);
    });

    it("records both operands as parents", () => {
      const a = new Value(10);
      const b = new Value(2);
      const c = a.div(b);
      expect(c.parents).toContain(a);
      expect(c.parents).toContain(b);
    });

    it("backprops correct gradients (quotient rule)", () => {
      // f = a / b
      // df/da = 1/b
      // df/db = -a / b^2
      const a = new Value(6);
      const b = new Value(3);
      const f = a.div(b);
      f.backward();

      expect(a.grad).toBeCloseTo(1 / 3); // 1/b = 1/3
      expect(b.grad).toBeCloseTo(-6 / (3 * 3)); // -a/b^2 = -6/9 = -0.6667
    });

    it("gradient of numerator is 1/denominator", () => {
      // isolate da: f = a / 4, df/da = 1/4 = 0.25
      const a = new Value(8);
      const b = new Value(4);
      a.div(b).backward();
      expect(a.grad).toBeCloseTo(0.25);
    });

    it("gradient of denominator is negative (-a/b^2)", () => {
      // f = 5 / b at b=2: df/db = -5/4 = -1.25
      const a = new Value(5);
      const b = new Value(2);
      a.div(b).backward();
      expect(b.grad).toBeCloseTo(-1.25);
    });

    it("composes in the graph: (a / b) * c", () => {
      // f = (a/b) * c
      // df/da = c/b
      const a = new Value(6);
      const b = new Value(2);
      const c = new Value(5);
      const f = a.div(b).mul(c);
      f.backward();

      expect(f.value).toBe(15); // (6/2)*5
      expect(a.grad).toBeCloseTo(5 / 2); // c/b = 2.5
    });
  });

  describe("backward — chain rule & graph behavior", () => {
    it("seeds the output gradient to 1", () => {
      const x = new Value(3);
      const out = x.add(new Value(2));
      out.backward();
      expect(out.grad).toBe(1);
    });

    it("applies the chain rule through composed ops", () => {
      // f = (a * b) + c
      // df/da = b, df/db = a, df/dc = 1
      const a = new Value(2);
      const b = new Value(3);
      const c = new Value(4);
      const f = a.mul(b).add(c);
      f.backward();
      expect(f.value).toBe(10); // 2*3 + 4
      expect(a.grad).toBe(3); // b
      expect(b.grad).toBe(2); // a
      expect(c.grad).toBe(1);
    });

    it("accumulates gradients when a value is used multiple times (x + x)", () => {
      // f = x + x = 2x, df/dx = 2  (NOT 1 — this is why += matters)
      const x = new Value(5);
      const f = x.add(x);
      f.backward();
      expect(f.value).toBe(10);
      expect(x.grad).toBe(2);
    });

    it("accumulates correctly for x * x (x squared)", () => {
      // f = x * x = x^2, df/dx = 2x = 2*3 = 6
      const x = new Value(3);
      const f = x.mul(x);
      f.backward();
      expect(f.value).toBe(9);
      expect(x.grad).toBe(6); // 2x — proves shared-node accumulation
    });

    it("handles a deeper expression: (x*x + x) ", () => {
      // f = x^2 + x, df/dx = 2x + 1 = 2*2 + 1 = 5
      const x = new Value(2);
      const f = x.mul(x).add(x);
      f.backward();
      expect(f.value).toBe(6); // 4 + 2
      expect(x.grad).toBe(5); // 2x + 1
    });

    it("composes tanh with arithmetic", () => {
      // f = tanh(a * b),  df/da = (1 - tanh^2(a*b)) * b
      const a = new Value(0.5);
      const b = new Value(2);
      const z = a.mul(b); // 1.0
      const f = z.tanh();
      f.backward();
      const localGrad = 1 - Math.tanh(1.0) ** 2;
      expect(a.grad).toBeCloseTo(localGrad * 2); // * b
      expect(b.grad).toBeCloseTo(localGrad * 0.5); // * a
    });

    it("does not duplicate a shared node in the toposort (visited works)", () => {
      // y = x*x; f = y + y  → x used through a shared intermediate
      // f = 2*(x^2), df/dx = 4x = 4*3 = 12
      const x = new Value(3);
      const y = x.mul(x);
      const f = y.add(y);
      f.backward();
      expect(f.value).toBe(18); // 2 * 9
      expect(x.grad).toBe(12); // 4x
    });
  });

  describe("Value.sub", () => {
    it("computes the difference", () => {
      expect(new Value(10).sub(new Value(3)).value).toBe(7);
      expect(new Value(2).sub(new Value(5)).value).toBe(-3);
    });

    it("backprops +1 to the first, -1 to the second", () => {
      // f = a - b ; df/da = 1, df/db = -1
      const a = new Value(10);
      const b = new Value(4);
      const f = a.sub(b);
      f.backward();
      expect(a.grad).toBe(1);
      expect(b.grad).toBe(-1);
    });

    it("composes in the graph", () => {
      // f = (a - b) * c ; df/da = c, df/db = -c
      const a = new Value(5);
      const b = new Value(2);
      const c = new Value(3);
      const f = a.sub(b).mul(c);
      f.backward();
      expect(f.value).toBe(9); // (5-2)*3
      expect(a.grad).toBe(3); // c
      expect(b.grad).toBe(-3); // -c
    });
  });

  describe("Value.pow", () => {
    it("computes the power", () => {
      expect(new Value(2).pow(3).value).toBe(8); // 2^3
      expect(new Value(9).pow(0.5).value).toBe(3); // sqrt(9)
    });

    it("backprops the power rule (n * x^(n-1))", () => {
      // f = x^3 ; df/dx = 3x^2 = 3*(2^2) = 12
      const x = new Value(2);
      const f = x.pow(3);
      f.backward();
      expect(x.grad).toBe(12);
    });

    it("sqrt gradient: d/dx[x^0.5] = 0.5 * x^(-0.5)", () => {
      // x = 4 ; grad = 0.5 * 4^(-0.5) = 0.5 * 0.5 = 0.25
      const x = new Value(4);
      const f = x.pow(0.5);
      f.backward();
      expect(x.grad).toBeCloseTo(0.25);
    });

    it("composes in the graph", () => {
      // f = (x^2) * 3 ; df/dx = 2x * 3 = 6x = 6*5 = 30
      const x = new Value(5);
      const f = x.pow(2).mul(new Value(3));
      f.backward();
      expect(f.value).toBe(75); // 25 * 3
      expect(x.grad).toBe(30); // 6x
    });
  });

  describe("Value.log", () => {
    it("computes natural log", () => {
      expect(new Value(1).log().value).toBe(0); // ln(1) = 0
      expect(new Value(Math.E).log().value).toBeCloseTo(1); // ln(e) = 1
    });

    it("computes log of other values", () => {
      expect(new Value(10).log().value).toBeCloseTo(Math.log(10));
    });

    it("backprops gradient of 1/x", () => {
      // d/dx ln(x) = 1/x ; at x=4, grad = 0.25
      const x = new Value(4);
      const out = x.log();
      out.backward();
      expect(x.grad).toBeCloseTo(0.25);
    });

    it("composes in the graph", () => {
      // f = ln(x) * 3 ; df/dx = 3/x = 3/2 = 1.5
      const x = new Value(2);
      const f = x.log().mul(new Value(3));
      f.backward();
      expect(x.grad).toBeCloseTo(1.5); // 3/x
    });

    it("inverse relationship with exp (sanity)", () => {
      // ln(e^x) should ≈ x
      const x = new Value(2);
      const out = x.exp().log();
      expect(out.value).toBeCloseTo(2);
    });
  });
});
