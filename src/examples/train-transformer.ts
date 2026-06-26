import { Transformer } from "../layers";
import { crossEntropyLoss } from "../operations";

const train = (
  // input
  text: string = "hello world",

  // Model Tuning
  dimension: number = 16,
  numHeads: number = 2,
  numBlocks: number = 2,

  // Learning Parameters
  learingRate: number = 0.1,
  steps: number = 100,
) => {
  // Create each letter vocabulary
  const vocabulary = [...new Set(text)].sort();
  const vocabSize = vocabulary.length;

  // Lookup maps for both id to char and char to id
  const charToId = new Map(vocabulary.map((vocab, index) => [vocab, index]));

  // Encode the whole text to ids, basically we are generating, id
  // correspondance of the given text, like how would you write Hello World in
  // the ids that we have generated
  const encoded = [...text].map((char) => charToId.get(char)!);

  // PrepareInput and TargetId, for each input token, the output will be the
  // next token of the encoded
  const inputIds = encoded.slice(0, -1); // From start to the end - 1, as we have the predict the last token
  const targetIds = encoded.slice(1); // All ids after 1

  /* ------------- Prepare Model ------------------- */
  const maxSeqLength = inputIds.length;
  const model = new Transformer(vocabSize, dimension, numHeads, numBlocks, maxSeqLength);

  /* ------------ Training Loop --------------------- */
  for (let step = 0; step < steps; step++) {
    // Initialize each parameter to zero
    for (const p of model.parameters()) {
      p.grad = 0;
    }

    const logits = model.forward(inputIds);
    const loss = crossEntropyLoss(logits, targetIds);

    loss.backward();

    for (const p of model.parameters()) {
      p.value -= learingRate * p.grad;
    }

    console.log(`Step ${step}: loss = ${loss.value.toFixed(4)}`);
  }
};

train();
