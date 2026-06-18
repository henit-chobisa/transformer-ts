import { Value } from "../grad";

export const softmax = (a: Value[]): Value[] => {
  const sum: Value = a.reduce((prev: Value, current: Value) => {
    return prev.add(current.exp());
  }, new Value(0));

  return a.map((value) => {
    return value.exp().div(sum);
  });
};
