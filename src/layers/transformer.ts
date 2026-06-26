import type { Value } from "../grad";
import { matmul } from "../operations";
import { addMatrix } from "../operations/add";
import { initializeRandomMatrix } from "../utils";
import { Embedding } from "./embedding";
import { PositionalEncoding } from "./encoding";
import { TransformerBlock } from "./transformer-block";

export class Transformer {
  embedding: Embedding;
  positional: PositionalEncoding;
  blocks: TransformerBlock[];

  /*
   * As we are going to need, the final output as the vocabSize x dimension,
   * which basically is the way to figure out, what is the probability for this
   * word to come next, that's what we are doing, these are the weights for
   * that.
   */
  finalProjection: Value[][];

  constructor(
    vocabSize: number,
    dimension: number,
    numHeads: number,
    numBlocks: number,
    maxSeqLength: number,
  ) {
    this.embedding = new Embedding(vocabSize, dimension);
    this.positional = new PositionalEncoding(maxSeqLength, dimension);
    this.blocks = Array.from(
      { length: numBlocks },
      () => new TransformerBlock(dimension, numHeads),
    );

    this.finalProjection = initializeRandomMatrix(dimension, vocabSize);
  }

  forward(tokenIds: number[]): Value[][] {
    // Getting the input ready, from words to vectors
    const inputEmbedding = this.embedding.forward(tokenIds);
    const positionalEncoding = this.positional.forward(tokenIds.length);
    let embedding = addMatrix(inputEmbedding, positionalEncoding);

    // Now for each transformer block we are going to run the forward
    embedding = this.blocks.reduce((acc, block) => block.forward(acc), embedding);

    // Calculating logits
    const logits = matmul(embedding, this.finalProjection);
    return logits;
  }

  parameters(): Value[] {
    return [
      ...this.embedding.parameters(),
      ...this.positional.parameters(),
      ...this.blocks.flatMap((block) => block.parameters()),
      ...this.finalProjection.flat(),
    ];
  }
}
