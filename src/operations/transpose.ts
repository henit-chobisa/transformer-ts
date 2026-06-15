import type { Value } from "../grad";

export function transpose(a: Value[][]): Value[][] {
  const transposed: Value[][] = new Array();

  a.forEach((rowValues: Value[]) => {
    rowValues.forEach((rowValue, indexCol) => {
      if (!transposed[indexCol]) {
        transposed[indexCol] = new Array();
      }

      transposed[indexCol].push(rowValue);
    });
  });

  return transposed;
}
