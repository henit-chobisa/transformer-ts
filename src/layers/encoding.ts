import type { Value } from "../grad";
import { initializeRandomMatrix } from "../utils";
/*
 * Of course you'd know about it, man bites dog, is gonna be more funny than dog
 * bites man, the thing is that position matters, and there is no layer that
 * actually respects the position of the word, if you look at attention, it just
 * sees a bag of words, and just simply dot product it, it doesn't care about
 * the position at all, that makes us take resposibility of the position and
 * that's why we are making this new implemention that is going to provide us
 * with a new matrix, which has positional vector.
 */
export class PositionalEncoding {
  positionTable: Value[][];

  constructor(maxSeqLen: number, dimension: number) {
    this.positionTable = initializeRandomMatrix(maxSeqLen, dimension);
  }

  forward(seqLength: number): Value[][] {
    if (seqLength > this.positionTable.length) {
      throw new Error("[Assertion Failed] seqLength greater than maxSeqLength");
    }

    return this.positionTable.slice(0, seqLength);
  }

  parameters(): Value[] {
    return this.positionTable.flat();
  }
}
