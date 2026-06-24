import { Value } from "../grad";
import { matmul } from "../operations";
import { softmax } from "../operations/softmax";
import { transpose } from "../operations/transpose";
import { initializeRandomMatrix } from "../utils";

type TQKVContainer = { Q: Value[][]; K: Value[][]; V: Value[][] };

/*
 * A single head attention, a simple class, that applies your QKV,
 * for the inputs given to us, using the matmul and other operations
 * that we have created. One attention head, one pattern
 */
export class Attention {
  wq: Value[][];
  wk: Value[][];
  wv: Value[][];

  constructor(dimension: number) {
    this.wq = initializeRandomMatrix(dimension, dimension);
    this.wk = initializeRandomMatrix(dimension, dimension);
    this.wv = initializeRandomMatrix(dimension, dimension);
  }

  /*
   * forward method, performs calculates, QKV for each of the present
   * vectors, and makes them context aware, when the given words Q, and
   * the remaining words K, which is going to give us the final context
   * aware weight, now with V, we are going to provide the actaul value
   */
  forward(input: Value[][]): Value[][] {
    // Each matrix container, each embedding's Q, K, and V
    const { Q, K, V } = this.generateQKV(input);
    const scores = matmul(Q, transpose(K));
    // A percentage of proportion of how much each feature weigh, and we are
    // going to take the value in proportion of it, the pillar for attentionm
    const contextWeights = scores.map((embeddingScore) => softmax(embeddingScore));

    /*
     * Now we have the context weight, a number per word word pair, that holds
     * context, which is similar to what we would have done, embedding by
     * embedding, but the only difference here is that, we are doing it,
     * in a full batch and each position serves it's own role
     */

    return matmul(contextWeights, V);
  }

  generateQKV(embedding: Value[][]): TQKVContainer {
    const Q = matmul(embedding, this.wq);
    const K = matmul(embedding, this.wk);
    const V = matmul(embedding, this.wv);

    return {
      Q,
      K,
      V,
    };
  }
}
