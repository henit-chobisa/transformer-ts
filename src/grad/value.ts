import { EOPERATIONTYPE } from "../types";

/*
 * An object that represents and holds a number, and where it came from
 * Such that if we know what operation it came from, and who were the
 * parents, we can calculate the gradient, easily. That's what the whole
 * point why we cannot use a number directly, and why having a value class
 * helps
 */
export class Value {
  // Actual value
  value: number;

  // Operands
  parents: Value[] = [];
  operation: EOPERATIONTYPE | undefined;

  // Gradient
  grad: number = 0;

  /*
   * The idea of this is that, we can connect the return value back
   * to the parents, like if you imagine, say we return a value, how
   * can we connect it back to the computation graph, for example, say
   * that we return, z = x + y, when we will try to calculate the
   * gradient, we will have to connect the z back to x and y, so that
   * we can calculate the gradient of x and y
   */
  _backward: () => void = () => {};

  constructor(value: number, parents: Value[] = [], operation?: EOPERATIONTYPE) {
    this.value = value;
    this.parents = parents;
    this.operation = operation;
  }

  /*
   * A simple multiplication operation, that will return a new value, not the
   * complicated part, the grad is worth explaining, so whenever we take a
   * differntial of 2*y, and we have a value of 2, that means y (the variable),
   * resolves to 1, giving us the value 2. So in each case, for this.grad, this
   * is "the variable", so the grad is to's value, and to's grad is this.value
   * multiplied by the grad of this.value, and the multiplication of result.grad
   * is chain rule
   */
  mul(to: Value): Value {
    const newValue = to.value * this.value;
    /*
     * So here in the backward sweep, we are trying to apply the chain rule to
     * calculate the gradient, at the current node, and the parent node. Say we
     * have w1, w2, and w3, then in that case, to land to w2, we are gonna have
     * to multiply with w3. A better example would be, z = (2x + 5) + 3x, in
     * that case, say we split this up into two functions, and call, y = 2x + 5,
     * in that case, we have z = y + 3x. Now we have the gradient for z, now
     * we move on, get the gradient of z, then we take the gradient of y wrt x,
     * and then multiply them to get the actual gradient. That's why we need to
     * get hold of the result z, and write a backward function, to assign the
     * right grad
     */
    const result = new Value(newValue, [this, to], EOPERATIONTYPE.MUL);
    result._backward = () => {
      /*
       * Here we have the `+=`, we have `=` that is alright, and make sense,
       * but in cases where the value is used twice, like x^2 that's not true
       * for us. That's not correct and that's the reason we need to do that.
       */
      this.grad += to.value * result.grad; // the downward propogating gradient
      to.grad += this.value * result.grad; // the downward propogating gradient
    };

    return result;
  }

  /*
   * Similary imagine, we are taking a differential of 1 + y, then 1, resolves
   * to 0, and the variable resolves to 1, so in this case, the gradient
   * resolves to 1, explaining further on the below operation, we are taking,
   * result.grad * (grad(1 + y)) => result.grad * (0 + 1) => result.grad * 1
   */
  add(to: Value): Value {
    const newValue = to.value + this.value;
    const result = new Value(newValue, [this, to], EOPERATIONTYPE.ADD);

    result._backward = () => {
      to.grad += result.grad * 1;
      this.grad += result.grad * 1;
    };

    return result;
  }

  /*
   * The almighty -> Creator of possibitities and flixibility of our straight
   * line like lives, here is the activiation function
   * Well I remember this from my high school, the differntial of tanh(x)
   * resolves to 1 - tanh(x)^2,
   */
  tanh(): Value {
    const value = Math.tanh(this.value);
    const result = new Value(value, [this], EOPERATIONTYPE.TANH);
    result._backward = () => {
      this.grad += (1 - result.value * result.value) * result.grad;
    };

    return result;
  }

  exp(): Value {
    const value = Math.exp(this.value);
    const result = new Value(value, [this], EOPERATIONTYPE.EXP);
    result._backward = () => {
      this.grad += result.value * result.grad;
    };

    return result;
  }

  div(to: Value): Value {
    const value = this.value / to.value;
    const result = new Value(value, [this, to], EOPERATIONTYPE.DIV);

    result._backward = () => {
      this.grad += (1 / to.value) * result.grad;
      to.grad += (-this.value / (to.value * to.value)) * result.grad;
    };

    return result;
  }

  /*
   * See before you read this function, the reason why it looks the way it looks
   * because of one simple fact, say that you defined `const two = new Value(2)`,
   * the thing is that, this two, can be used in any layer to be honest, it's a
   * number, I can use it for square with the same, that will have the same
   * object reference twice, and for the same reason, we need to perform a
   * topological sort, such that we make sure that every node
   * resolves it's gradient before it's parents as if you notice your
   * _backward function, it needs the parent's gradient to be resolved. The
   * below code becomes self-explanatory, as simple as we are only going to push
   * a node, once we know that all it's parents and ancestors are visited.
   * That's all the concept is. And then we are going to start from the child,
   * to the parent, resolving the gradient, that's the reason we need reversing
   */
  backward(): void {
    const topo: Value[] = [];
    const visited = new Set<Value>();
    const build = (node: Value): void => {
      if (visited.has(node)) return;
      visited.add(node);
      for (const parent of node.parents) {
        build(parent);
      }
      topo.push(node);
    };
    build(this);
    this.grad = 1;
    for (let i = topo.length - 1; i >= 0; i--) {
      topo[i]?._backward();
    }
  }
}
