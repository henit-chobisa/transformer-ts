import { Value } from "../grad";
import { softmax } from "./softmax";

/*
 * For this setup, try and understand that, logits as a 2D matrix is not 1
 * token's logits, it's spread across, N x Vocabsize, bascially each row holds
 * probability of each vocabulary word to be next, and we need to calculate loss
 * of that only. Each target id is basically the token after which we are going
 * to predict the next token.
 * Now understand the shape of targetId array, each targetId value holds the
 * actual prediction, that should come, the token's id that should code, and the
 * position is the bridge between the logits and the targetIds.
 *
 * Reference Claude:
 * targetIds = [2, 0, 1]
             │  │  │
             │  │  └─ row 2's correct letter was #1 (b)
             │  └──── row 1's correct letter was #0 (a)
             └─────── row 0's correct letter was #2 (c)
 */
export const crossEntropyLoss = (logits: Value[][], targetIds: number[]): Value => {
  const allTokenLogitsLength = logits.length;

  const perPositionLoss = logits.map((logitForEachTarget, index) => {
    const probabilitiesOfEachVocabWord = softmax(logitForEachTarget);
    // The next token that should come
    const correctNextToken = targetIds[index];
    const probabilityOfCorrectTokenGivenByModel = probabilitiesOfEachVocabWord[correctNextToken!]!;

    // Now we can calculate the loss between these two values
    return probabilityOfCorrectTokenGivenByModel.log().mul(new Value(-1));
  });

  const sum = perPositionLoss.reduce((prev, next) => {
    return prev.add(next);
  }, new Value(0));

  return sum.div(new Value(allTokenLogitsLength));
};
