import type { Value } from "../grad";
import { dotProduct } from "./dot-product";
import { transpose } from "./transpose";

/*
 * matmul is your matrix multiplication actually, you multiply each position in
 * the row, to the column, that requies that, your alternate dimensions are
 * equal, and rest all each of it is the same dot product function that we just
 * do, each row with each column
 */
export function matmul(a: Value[][], b: Value[][]): Value[][] {
  // Safety checks
  const rowsWithUnmatchedAlternateDimension = a.filter((rows) => rows.length !== b.length);
  if (rowsWithUnmatchedAlternateDimension.length > 0) {
    throw new Error("[Assertion Failed], Alternate dimensions must match for matmul");
  }

  // Let's get the transpose of B, so things become easy, we have all values in
  // one place in that case
  const bTranspose = transpose(b);

  const result: Value[][] = new Array();

  a.forEach((row, rowIndex) => {
    bTranspose.forEach((column) => {
      if (!result[rowIndex]) result[rowIndex] = new Array();
      const resultValue = dotProduct(row, column);
      result[rowIndex].push(resultValue);
    });
  });

  return result;
}
