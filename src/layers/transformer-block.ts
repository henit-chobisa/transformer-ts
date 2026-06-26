import type { Value } from "../grad";
import { addMatrix } from "../operations/add";
import { layernorm } from "../operations/layernorm";
import { MLP } from "./mlp";
import { MultiHeadAttention } from "./multi-head-attention";

export class TransformerBlock {
  attention: MultiHeadAttention;
  mlp: MLP;

  constructor(dimension: number, heads: number) {
    this.attention = new MultiHeadAttention(dimension, heads);
    this.mlp = new MLP(dimension, [dimension, dimension]);
  }

  forward(inputs: Value[][]): Value[][] {
    // Let's first use attention and make out inputs context aware
    const attentionResidual = this.attention.forward(inputs);
    const contextAwareInputs = addMatrix(inputs, attentionResidual);
    const normalizedInputs = layernorm(contextAwareInputs);

    /*
     * Now our inputs are ready, all good. Now we can run MLP for each row, if
     * you have the batching picture in the mind, it's working a little
     * different than that, we are running MLP on each word, that means we are
     * going to predict each feature of the word as part of the output, not the
     * actual word as you expect.
     */
    const ffOut = normalizedInputs.map((vector) => this.mlp.forward(vector));
    const alteredOutputFeatures = addMatrix(normalizedInputs, ffOut);

    // Normalized outputs to consistent scale
    return layernorm(alteredOutputFeatures);
  }

  parameters(): Value[] {
    return [...this.attention.parameters(), ...this.mlp.parameters()];
  }
}
