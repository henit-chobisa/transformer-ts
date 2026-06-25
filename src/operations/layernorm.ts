import { Value } from "../grad";

const epsilon = new Value(1e-5);

export const layernorm = (matrix: Value[][]): Value[][] => {
  const matrixMean = matrix.map((row) => calculateMean(row));
  const matrixVariance = matrix.map((row, index) => calculateVariance(row, matrixMean[index]!));

  return matrix.map((row, rowIndex) => {
    const mean = matrixMean[rowIndex]!;
    const variance = matrixVariance[rowIndex]!;
    const dinominator = variance.add(epsilon).pow(0.5)!;

    return row.map((value) => value.sub(mean).div(dinominator));
  });
};

const calculateMean = (row: Value[]): Value => {
  const sum = row.reduce((prev, current) => {
    return prev.add(current);
  }, new Value(0));

  return sum.div(new Value(row.length));
};

const calculateVariance = (row: Value[], mean: Value): Value => {
  const sumOfSquaredDifference = row.reduce((prev, current) => {
    return prev.add(current.sub(mean).pow(2));
  }, new Value(0));

  return sumOfSquaredDifference.div(new Value(row.length));
};
