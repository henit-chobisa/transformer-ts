import { Value } from "../grad";

/*
 * Function takes two Value arrays, and the utility is say that, we have w1*x1 +
 * w2 * x2 + w3 * x3, that's what we are making here, but the output is required
 * to be a value class, that's needed, such that imagine result = act(z), where
 * z is our function, we need to apply the chain rule, and we wish to understand
 * gradient of each contributor, such that we can nudge them, and understand,
 * which one contirbutes to the lose. Basically gradient decent
 */
export const dotProduct = (a: Value[], b: Value[]): Value => {
  if (a.length !== b.length) {
    throw new Error("[Assertion Failed] The length of both participants should be equal");
  }

  /*
   * The two operations that we are going to perform, first is
   * a[i].mul(b[i]) and another one that we are going to do is result of them
   * added to the result of it .... wn*xn
   */

  return a.reduce((prev: Value, current: Value, index) => {
    const targetBValue = b[index];

    if (!targetBValue) {
      throw new Error(
        "[Assertion Failed] Every array value is expected to be defined, recieved undefined",
      );
    }

    return prev.add(current.mul(targetBValue));
  }, new Value(0));
};
