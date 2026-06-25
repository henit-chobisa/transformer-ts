import { Value } from "../grad/value.js";

export class Neuron {
  weights: Value[] = [];
  bias: Value = new Value(0);

  constructor(numInputs: number) {
    // We'll randomly initialize each weight, from -1 to 1, as we are using tanh
    // activation function
    let count = 0;
    while (count < numInputs) {
      const randomNumber = Math.random() * 2 - 1;
      this.weights.push(new Value(randomNumber));
      count++;
    }

    this.bias = new Value(Math.random() * 2 - 1);
  }

  forward(inputs: Value[]): Value {
    if (inputs.length !== this.weights.length) {
      throw new Error("Assertion failed, inputs should be equal to the weights");
    }

    const preActivationResult = inputs.reduce((previousValue, currentValue, index): Value => {
      const weightValue = this.weights[index]!;
      return previousValue.add(currentValue.mul(weightValue));
    }, this.bias);

    return preActivationResult.tanh();
  }

  parameters(): Value[] {
    return [...this.weights, this.bias];
  }
}
