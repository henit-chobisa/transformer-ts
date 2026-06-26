import { Value } from "../grad";
import { initializeRandomMatrix } from "../utils";

/*
 * This is the feeder class, basically we are going to initialize a random set
 * of matrix, where each row is the embedding, that we are assuming is mapped to
 * a word, basically, we are starting completely random, and we are going to
 * train according to the mapping of the word.
 */
export class Embedding {
  vocabularyVectors: Value[][];

  constructor(vocabSize: number, dimension: number) {
    this.vocabularyVectors = initializeRandomMatrix(vocabSize, dimension);
  }

  // The ids of the rows that you're going to need, we are going to return a new
  // matrix, with all those token embeddings
  forward(tokenIds: number[]): Value[][] {
    const maxLengthToken = Math.max(...tokenIds);
    if (maxLengthToken >= this.vocabularyVectors.length) {
      throw new Error("[Assertion Failed] Token ID out of bounds");
    }

    return tokenIds.map((tokenId) => this.vocabularyVectors[tokenId]!);
  }

  parameters(): Value[] {
    return this.vocabularyVectors.flat();
  }
}
