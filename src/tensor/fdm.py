"""Module containing function(s) used for computing gradient using finite difference
    method.
"""
from collections.abc import Callable
from itertools import product
import numpy as np

def _ei(shape: tuple[int, ...], index: tuple[int, ...], val: float) -> np.ndarray:
    """Generate a zero ndarray of shape `shape` and set value `val` at appropriate
    `index`.

    Args:
        shape: shape of which a zero vector is generated
        index: index at which to set the value `val`
        val: value to set at index
    Returns:
        ndarray
    """
    x = np.zeros(shape, dtype=np.float64)
    x[index] = val
    return x

def FDM(f: Callable, inputs: list[np.ndarray], h: float=1e-5) -> list[np.ndarray]:
    """Compute the gradient vector of f with respect to inputs using finite difference
        method.

    Note:
        Finite difference method used here is central difference.
        f' = f(a + h) - f(a - h) / (2 * h)

    Example:
        > a = np.array([1, 2])
        > b = np.array([3, 4])
        > da, db, = FDM(lambda x, y: (x+y).sum(), [a, b])

    Args:
        f: function of which the gradient needs to be computed
        inputs: inputs of the function
        h: really small value
    Returns:
        list of ndarrays containing the partial derivative with respect to each input
    """
    grads = []
    lhs_inputs = [np.array(x, dtype=np.float64) for x in inputs]
    rhs_inputs = [np.array(x, dtype=np.float64) for x in inputs]

    for i, var in enumerate(inputs):
        grad = np.zeros_like(var, dtype=np.float64)

        indices = product(*[range(var.shape[k]) for k in range(var.ndim)])
        for idx in indices:
            eps = _ei(var.shape, idx, h)

            lhs_inputs[i] += eps 
            rhs_inputs[i] -= eps
            grad[idx] = (f(*lhs_inputs) - f(*rhs_inputs)) / (2*h)
            lhs_inputs[i] -= eps
            rhs_inputs[i] += eps
        grads.append(grad)
    return grads
