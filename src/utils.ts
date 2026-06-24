import { Value } from "./grad";

export const initializeRandomMatrix = (x: number, y: number): Value[][] => {
  const result: Value[][] = new Array();

  for (let i = 0; i < x; i++) {
    for (let j = 0; j < y; j++) {
      if (!result[i]) result[i] = new Array();
      result[i]!.push(new Value(Math.random() * 2 - 1));
    }
  }

  return result;
};
