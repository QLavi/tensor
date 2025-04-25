"""Forward and backward passes of math function that Tensor supports.
"""
from functools import partial
from .core import Tensor, _BContext, optional_array, forward_all
import numpy as np

def _exp_F(ctx: _BContext, x: Tensor) -> Tensor:
    _x = np.exp(x.data)
    ctx.save_array(_x)
    return Tensor(_x)

def _exp_B(ctx: _BContext, g: np.ndarray) -> tuple[optional_array, ...]:
    x, = ctx.saved_arrays
    dx = g * x if ctx.input_req_grads[0] else None
    return (dx, )


def _log_F(ctx: _BContext, x: Tensor) -> Tensor:
    ctx.save_array(x.data)
    return Tensor(np.log(x.data))

def _log_B(ctx: _BContext, g: np.ndarray) -> tuple[optional_array, ...]:
    x, = ctx.saved_arrays
    dx = g / x if ctx.input_req_grads[0] else None
    return (dx, )


def _log2_F(ctx: _BContext, x: Tensor) -> Tensor:
    ctx.save_array(x.data)
    return Tensor(np.log2(x.data))

def _log2_B(ctx: _BContext, g: np.ndarray) -> tuple[optional_array, ...]:
    x, = ctx.saved_arrays
    dx = g / (x * np.log(2)) if ctx.input_req_grads[0] else None
    return (dx, )


def _sum_F(ctx: _BContext, x: Tensor, axis: int | None = None) -> Tensor:
    ctx.save_array(x.data)
    if axis is None:
        r = np.sum(x.data)
    else:
        r = np.sum(x.data, axis=axis, keepdims=True)
    return Tensor(r)

def _sum_B(ctx: _BContext, g: np.ndarray) -> tuple[optional_array, ...]:
    x, = ctx.saved_arrays
    dx = np.broadcast_to(g, x.shape) if ctx.input_req_grads[0] else None
    return (dx, )


def _abs_F(ctx: _BContext, a: Tensor) -> Tensor:
    ctx.save_array(a.data)
    return Tensor(np.fabs(a.data))

def _abs_B(ctx: _BContext, g: np.ndarray) -> tuple[optional_array, ...]:
    x, = ctx.saved_arrays
    dx = g * np.sign(x) if ctx.input_req_grads[0] else None
    return (dx, )

def _maximum_F(ctx: _BContext, a: Tensor, b: Tensor) -> Tensor:
    ctx.save_array(a.data)
    ctx.save_array(b.data)
    return Tensor(np.maximum(a.data, b.data))

def _maximum_B(ctx: _BContext, g: np.ndarray) -> tuple[optional_array, ...]:
    a, b, = ctx.saved_arrays
    da = g * 0.5 * (1 + np.sign(a - b)) if ctx.input_req_grads[0] else None
    db = g * 0.5 * (1 - np.sign(a - b)) if ctx.input_req_grads[1] else None
    return da, db


def _minimum_F(ctx: _BContext, a: Tensor, b: Tensor) -> Tensor:
    ctx.save_array(a.data)
    ctx.save_array(b.data)
    return Tensor(np.minimum(a.data, b.data))

def _minimum_B(ctx: _BContext, g: np.ndarray) -> tuple[optional_array, ...]:
    a, b, = ctx.saved_arrays
    da = g * 0.5 * (1 - np.sign(a - b)) if ctx.input_req_grads[0] else None
    db = g * 0.5 * (1 + np.sign(a - b)) if ctx.input_req_grads[1] else None
    return da, db

def _max(x: Tensor) -> Tensor:
    return Tensor(np.max(x.data))

exp  = partial(forward_all, _exp_F, _exp_B)
log  = partial(forward_all, _log_F, _log_B)
sum  = partial(forward_all, _sum_F, _sum_B)
abs  = partial(forward_all, _abs_F, _abs_B)
log2 = partial(forward_all, _log2_F, _log2_B)
maximum = partial(forward_all, _maximum_F, _maximum_B)
minimum = partial(forward_all, _minimum_F, _minimum_B)
max = _max
