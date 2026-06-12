import { Value } from "../grad/";
import { MLP } from "../network/";

// Dataset
const xs: number[][] = [
  [2.0, 3.0, -1.0],
  [3.0, -1.0, 0.5],
  [0.5, 1.0, 1.0],
  [1.0, 1.0, -1.0],
];
const ys: number[] = [1.0, -1.0, -1.0, 1.0]; // desired outputs

function computeLoss(mlp: MLP, xs: number[][], ys: number[]): Value {
  const predictions = xs.map((x) => {
    const inputs = x.map((x) => new Value(x));
    return mlp.forward(inputs)[0];
  });

  return predictions.reduce((acc, pred, i) => {
    const target = new Value(ys[i]!);
    const diff = pred!.add(target.mul(new Value(-1)));
    const sq = diff.mul(diff);
    return acc!.add(sq);
  }, new Value(0))!;
}

const mlp = new MLP(3, [4, 4, 1]);
const learningRate = 0.5;

for (let step = 0; step < 50; step++) {
  for (const p of mlp.parameters()) {
    p.grad = 0;
  }

  const loss = computeLoss(mlp, xs, ys);
  loss.backward();

  for (const p of mlp.parameters()) {
    p.value -= p.grad * learningRate;
  }

  console.log(`step ${step}: loss = ${loss.value.toFixed(4)}`);
}

// Final predictions
console.log("\nFinal predictions:");
xs.forEach((x, i) => {
  const inputs = x.map((v) => new Value(v));
  const pred = mlp.forward(inputs)[0]!.value;
  console.log(`  ${x} → predicted ${pred.toFixed(3)}, target ${ys[i]}`);
});
