import type { Value } from "../grad";

export const addMatrix = (a: Value[][], b: Value[][]): Value[][] => {
  const aColumnLength = a[0]?.length;
  const bColumnLength = b[0]?.length;
  const aRowLength = a.length;
  const bRowLength = b.length;

  if (aColumnLength !== bColumnLength || aRowLength !== bRowLength) {
    throw new Error(
      "[Assertion Failed] For the matrix passed, all dimensional length should be same",
    );
  }

  return a.map((row, rowIndex) => {
    const bRow = b[rowIndex]!;
    return row.map((value, colIndex) => value.add(bRow[colIndex]!));
  });
};
