import type { Value } from "../grad";

export const concat = (matrices: Value[][][]): Value[][] => {
  const rowCount = matrices[0]?.length;
  const initial: Value[][] = Array.from({ length: rowCount! }, () => []);
  return matrices.reduce((prev: Value[][], current: Value[][]) => {
    return concatTwo(prev, current);
  }, initial);
};

const concatTwo = (a: Value[][], b: Value[][]): Value[][] => {
  if (a.length !== b.length) {
    throw new Error("[Assertion Failed] Both participants should be equal dimension");
  }

  return a.map((row, index) => {
    if (!b[index]) {
      throw new Error("[Assertion Failed] Expected array, received undefined");
    }

    return [...row, ...b[index]];
  });
};
