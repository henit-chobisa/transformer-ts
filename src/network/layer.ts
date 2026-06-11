import type { Value } from "../grad/value.js";
import { Neuron } from "./neuron.js";

export class Layer {
  neurons: Neuron[] = [];

  constructor(numInputs: number, numNeurons: number) {
    let count = 0;
    while (count < numNeurons) {
      this.neurons.push(new Neuron(numInputs));
      count++;
    }
  }

  forward(inputs: Value[]): Value[] {
    return this.neurons.map((neuron) => {
      return neuron.forward(inputs);
    });
  }

  parameters(): Value[] {
    return this.neurons.flatMap((neuron) => neuron.parameters());
  }
}
