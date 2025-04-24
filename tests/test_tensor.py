import tensor as t
import tensor.math as tm
from tensor.fdm import FDM

import numpy as np

_T = lambda x: np.transpose(x, axes=(*range(0, x.ndim -2), -1, -2))
rng = np.random.default_rng()

bi_ops = [
    lambda a, b: a + b,
    lambda a, b: a - b,
    lambda a, b: a * b,
    lambda a, b: a / b,
    lambda a, b: a @ b,
]
un_ops = [
    lambda a: -a,
    lambda a:  a.T,
]
ma_un_ops = [
    (tm.exp, np.exp),
    (tm.log, np.log),
    (tm.sum, np.sum),
    (tm.log2, np.log2),
    (tm.max, np.max),
]
ma_bi_ops = [
    (tm.maximum, np.maximum),
    (tm.minimum, np.minimum),
]

def test_operations():
    na = np.array([[[1, 2], [3, 4]], [[1, 2], [3, 4]]])
    nb = np.array([[1, 2], [3, 4]])
    a = t.Tensor(na, req_grad=True)
    b = t.Tensor(nb, req_grad=True)

    for op in bi_ops:
        assert (op(a, b).data == op(na, nb)).all()
        assert (op(b, a).data == op(nb, na)).all()

    for op in bi_ops[:-1]:
        assert (op(a, -7).data == op(na, -7)).all()

    assert (un_ops[0](a).data == un_ops[0](na)).all()
    assert (un_ops[1](a).data == _T(na)).all()

    for top, nop in ma_un_ops:
        assert (top(a).data == nop(na)).all()
    assert (tm.sum(a, axis=1).data == np.sum(na, axis=1)).all()

    for top, nop in ma_bi_ops:
        assert (top(a, b).data == nop(na, nb)).all()

def test_backward():
    na = np.array([[[1, 2], [3, 4]], [[1, 2], [3, 4]]])
    nb = np.array([[1, 2], [3, 4]])
    a = t.Tensor(na, req_grad=True)
    b = t.Tensor(nb, req_grad=True)

    t.tape_clear()
    for op in bi_ops:
        r = tm.sum(op(a, b))
        r.backward()
        da, db = FDM(lambda x, y: np.sum(op(x, y)), [na, nb])
        assert np.allclose(a.grad, da)
        assert np.allclose(b.grad, db)

    for op in un_ops:
        r = tm.sum(op(a))
        r.backward()
        da = FDM(lambda x: np.sum(op(x)), [na])
        assert np.allclose(a.grad, da)
        assert np.allclose(b.grad, db)

    for op, np_op in ma_un_ops[:-1]:
        r = tm.sum(op(a))
        r.backward()
        da = FDM(lambda x: np.sum(np_op(x)), [na])
        assert np.allclose(a.grad, da)

    r = tm.sum(tm.sum(a, axis=1))
    r.backward()
    da = FDM(lambda x: np.sum(np.sum(x, axis=1)), [na])
    assert np.allclose(a.grad, da)

    for op, np_op in ma_bi_ops:
        r = tm.sum(op(a, b))
        r.backward()
        da, db, = FDM(lambda x, y: np.sum(np_op(x, y)), [na, nb])
        assert np.allclose(a.grad, da)
        assert np.allclose(b.grad, db)

def test_general_functions():
    def numerical_sqrt(x, N=4):
        t = (1 + x) / 2
        for _ in range(1, N):
            t = (t + x / t) / 2
        return t

    np_a = np.array([[[1, 2], [3, 4]], [[1, 2], [3, 4]]])
    a = t.Tensor(np_a, req_grad=True)

    r = tm.sum(numerical_sqrt(a, N=10))
    r.backward()

    np_numerical_sqrt = lambda x: np.sum(numerical_sqrt(x, N=10))
    da, = FDM(np_numerical_sqrt, [np_a])
    assert np.allclose(a.grad, da)

    np_a = rng.random((5, 7, 3, 6))
    np_b = rng.random((5, 7, 6, 3))
    np_c = rng.random((5, 7, 3, 3))
    r = (np_a @ np_b) @ np_c

    a = t.Tensor(np_a, req_grad=True)
    b = t.Tensor(np_b, req_grad=True)
    c = t.Tensor(np_c, req_grad=True)
    r = tm.sum((a @ b) @ c)
    r.backward()
    da, db, dc = FDM(lambda a, b, c: np.sum((a @ b) @ c), [np_a, np_b, np_c])
    assert np.allclose(a.grad, da)
    assert np.allclose(b.grad, db)
    assert np.allclose(c.grad, dc)

    def np_softmax(x):
        e_x = np.exp(x)
        return e_x / np.sum(e_x, axis=1, keepdims=True)

    def tm_softmax(x):
        e_x = tm.exp(x)
        return e_x / tm.sum(e_x, axis=1)

    np_a = np.array([[1, 2, 3], [3, 4, 5]], dtype=np.float64)
    a = t.Tensor(np_a, req_grad=True)
    assert np.allclose(tm_softmax(a).data, np_softmax(np_a))

    t.tape_clear()
    r = tm.sum(tm_softmax(a))
    r.backward()
    da, = FDM(lambda a: np.sum(np_softmax(a)), [np_a])
    assert np.allclose(a.grad, da)

    def CCE(q, p):
        return -tm.sum(p * tm.log(q)) / q.shape[0]

    def np_CCE(q, p):
        return -np.sum(p * np.log(q)) / q.shape[0]

    np_pred = np.array([[0.6, 0.39, 0.01], [0.3, 0.3, 0.4]])
    np_prob = np.array([[1, 0, 0], [0, 0, 1]])
    pred = t.Tensor(np_pred, req_grad=True)
    prob = t.Tensor(np_prob)

    r = CCE(pred, prob)
    r.backward()
    da, db, = FDM(lambda q, p: np_CCE(q, p), [np_pred, np_prob])
    assert np.allclose(pred.grad, da)
