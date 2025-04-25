import math
from functools import partial
import numpy as np

import tensor as t
import tensor.math as tm
from tensor.fdm import FDM

rng = np.random.default_rng(42)

np_Ws = [
    rng.random((8, 1)),
    rng.random((4, 8)),
    rng.random((1, 4)),
]
np_bs = [
    rng.random((8, 1)),
    rng.random((4, 1)),
    rng.random((1, 1)),
]

tm_Ws = [t.Tensor(np_W, req_grad=True) for np_W in np_Ws]
tm_bs = [t.Tensor(np_b, req_grad=True) for np_b in np_bs]

def tm_F(M):
    M = M.T
    M = tm_sigmoid(tm_Ws[0] @ M + tm_bs[0])
    M = tm_sigmoid(tm_Ws[1] @ M + tm_bs[1])
    M = tm_Ws[2] @ M + tm_bs[2]
    M = M.T
    return M

def np_F(M, W0, b0, W1, b1, W2, b2):
    M = M.T
    M = np_sigmoid(W0 @ M + b0)
    M = np_sigmoid(W1 @ M + b1)
    M = W2 @ M + b2
    M = M.T
    return M

def tm_sigmoid(x): return 1 / (1 + tm.exp(-x))
def tm_relu(x): return tm.maximum(t.zeros_like(x), x)

def np_sigmoid(x): return 1 / (1 + np.exp(-x))
def np_relu(x): return np.maximum(np.zeros_like(x), x)

def tm_MSE(a, b):
    d = a - b
    return tm.sum(d*d) / a.shape[0]

def np_MSE(a, b):
    d = a - b
    return np.sum(d*d) / a.shape[0]

def test_neuralnet():
    np_X = rng.random((10, 1))
    np_Y = np.sin(np_X)

    X = t.Tensor(np_X)
    Y = t.Tensor(np_Y)
    r = tm_MSE(tm_F(X), Y)
    r.backward()

    def new_F(W0, b0, W1, b1, W2, b2):
        nonlocal np_X, np_Y
        return np_MSE(np_F(np_X, W0, b0, W1, b1, W2, b2), np_Y) 

    dW0, db0, dW1, db1, dW2, db2 = FDM(new_F, [
        np_Ws[0], np_bs[0], 
        np_Ws[1], np_bs[1],
        np_Ws[2], np_bs[2]
    ])
    dWs = [dW0, dW1, dW2]
    dbs = [db0, db1, db2]

    for i in range(3):
        assert np.allclose(tm_Ws[i].grad, dWs[i])
        assert np.allclose(tm_bs[i].grad, dbs[i])
