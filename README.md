# tensor

A reverse mode automatic differentiation library.

After being tired of writing and debugging backpropogration algorithm for neural nets.
I implemented this so algorithms for computing gradients and algorithms for neural network are kept separate.

## Features:

- Supports numpy like broadcasting.
- Some of numpy's math functions(log, exp, so on).

## Install

Please install this in a separte virtual environment
In the source directory of the project run `pip3 install .`

## Example

```py
import tensor as t
import tensor.math as tm

a = t.Tensor([[1, 2], [3, 4]], req_grad=True)
b = t.Tensor([[1, 2]], req_grad=True)
c = tm.sum(3 * a + -b)
c.backward()

print(a.grad) # [[3, 3], [3, 3]]
print(b.grad) # [[-2, -2]]
```

## Resources from which I drew relevant information from:

- Pytorch Documentation
- Articles written by Eli Bendersky
