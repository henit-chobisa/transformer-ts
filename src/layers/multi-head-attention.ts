import type { Value } from "../grad";
import { matmul } from "../operations";
import { concat } from "../operations/concat";
import { initializeRandomMatrix } from "../utils";
import { Attention } from "./attention";

export class MultiHeadAttention {
  heads: Attention[];
  wo: Value[][];

  /*
   * Dimension can be understood as the number of features that we will have in
   * one input block, and heads as the number of attention heads that we have at
   * this point. Eact Attention Head = One Pattern
   */
  constructor(dimension: number, heads: number) {
    if (dimension % heads !== 0) {
      throw new Error("[Assertion Failed] dimension must be divisible by number of heads");
    }

    const headDimension = dimension / heads;
    // Initialize heads
    this.heads = Array.from({ length: heads }, () => new Attention(dimension, headDimension));
    this.wo = initializeRandomMatrix(dimension, dimension);
  }

  forward(input: Value[][]): Value[][] {
    const perHeadContextAwareInputs = this.heads.map((head) => head.forward(input));
    // The dimension equals to the actual input given to us
    const concatanatedContextAwareInputs = concat(perHeadContextAwareInputs);
    const normalizedInputs = matmul(concatanatedContextAwareInputs, this.wo);
    return normalizedInputs;
  }

  parameters(): Value[] {
    return [...this.heads.flatMap((head) => head.parameters()), ...this.wo.flat()];
  }
}
