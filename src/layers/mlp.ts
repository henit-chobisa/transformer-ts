import type { Value } from "../grad/value.js";
import { Layer } from "./layer.js";

export class MLP {
  layers: Layer[];
  constructor(numInputs: number, layerSizes: number[]) {
    const sizes = [numInputs, ...layerSizes]; // [3, 4, 4, 1]
    this.layers = layerSizes.map((_, i) => new Layer(sizes[i]!, sizes[i + 1]!));
  }

  forward(inputs: Value[]): Value[] {
    return this.layers.reduce((acc, layer) => layer.forward(acc), inputs);
  }

  parameters(): Value[] {
    return this.layers.flatMap((neuron) => neuron.parameters());
  }
}
