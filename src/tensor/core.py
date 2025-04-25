"""Contains classes used in automatic-differentiation(autodiff) engine. """
from dataclasses import dataclass, field
from collections.abc import Callable
import typing as ty
from functools import partial
import numpy as np

optional_array = np.ndarray | None
_rng = np.random.default_rng()

_tape: list['_TapeItem'] = []
_T = lambda x: np.transpose(x, axes=(*range(0, x.ndim -2), -1, -2))

@dataclass
class _BContext:
    """Stores ndarrays used during the forward pass of the computation graph.

    Attributes:
        input_reg_grads: tuple of booleans used to check whether or 
            not to compute the gradient.
        saved_arrays: list storing the ndarrays saved during the forward pass.
    """
    input_req_grads: tuple[bool, ...] = field(default_factory=tuple, init=False)
    saved_arrays: list[np.ndarray] = field(default_factory=list, init=False)

    def save_array(self, arr: np.ndarray):
        """Saves the given ndarray.

        Args:
            arr: ndarray to save.
        Returns:
            None
        """
        self.saved_arrays.append(arr)


@dataclass(eq=False)
class _TapeItem:
    """Stores the information required for the backward pass.

    For computing the gradient of function. We store the function that is needed
    to be called for computing the gradient of that function (backward). Backward.
    needs output of the function for propogating the gradient from output to its inputs.

    Attributes:
        backward: function that needs to be called to run the backward pass for
            operation or function.
        ctx: stores ndarrays saved during the forward pass.
        inputs: input tensors for the forward pass.
        output: output tensors of the forward pass.
    """
    backward: Callable
    ctx: _BContext = field(repr=False)
    inputs: tuple['Tensor', ...] = field(repr=False)
    output: 'Tensor' = field(repr=False)


def forward_all(F: Callable, B: Callable, *args, **kwargs) -> 'Tensor':
    """A Function wrapper that saves information in relevant objects.
    Args:
        F: forward pass function
        B: backward pass function
        *args, **kwargs: arguments relevant to F
    Returns:
        output tensor of F
    """
    args_proper = tuple(
        x if isinstance(x, Tensor) else Tensor(x)
        for x in args
    )
    ctx = _BContext()
    r = F(ctx, *args_proper, **kwargs)
    ctx.input_req_grads = tuple(x.req_grad for x in args_proper)
    r.req_grad = any(ctx.input_req_grads)
    if r.req_grad:
        _tape.append(_TapeItem(B, ctx, args_proper, r))
    return r


@dataclass(init=False, eq=False)
class Tensor:
    """Stores the data, gradient ndarrays, and tracks the operation performed on
        the data.

    When operation from the tmath module is used on the Tensor instance. it saves
    information required to compute the gradient of the operation with respect to
    all its inputs.

    Note:
        - Operation named _op_F and _op_B are the forward and backward passes of 
        elementary operations for which this classes supports operator overloading.

        - This class cannot support operator overload in which ndarrays are involved
        __rop__ of this class will never be called. __op__ of numpy ndarray works in
        two steps:
            1. Try performing the operation on ndarray and the class.
            2. If that fails try performing the operation with ndarrays' elements
                and the class, and this will promote to resulting ndarray 
                with dtype `object`. If class implements operator overloading 
                with ndarray's element's type.
    Example:
        import tensor as t
        > x = t.Tensor(np.array([[1, 2], [3, 4]]), req_grad=True)
        > x += 29
        > x.backward()
        > x.grad
          [[1, 1], [1, 1]]

    Attributes:
        data: ndarray storing the data of the tensor.
        grad: gradient of the operation with respect to this instance.
        req_grad: flag used to check whether the tensor is variable or a constant.
    """
    def __init__(self,
        data: np.ndarray | int | float,
        req_grad: bool = False):
        """
        Args:
            data: data that tensor will store
            req_grad: whether this tensor requires gradient of functions it get used
                in.
        Note:
            Using 1d ndarray is problematic because of how it broadcasts, an example
            for this is,
            M = np.array([[1, 2], [3, 4]])
            x = np.array([1, 2])
            r = M @ x # here x will broadcast to shape (2, 1)
            r = x @ M # here x will broadcast to shape (1, 2)
            so self.data's ndim should be atleast 2d.
        """

        if isinstance(data, np.ndarray) and data.ndim == 1:
            self.data = np.atleast_2d(data)
        else:
            self.data = np.array(data)

        self.data = self.data.astype(np.float64)
        self.req_grad = req_grad
        self.grad = None

    def __getitem__(self, index) -> 'Tensor':
        """Index in the tensor's data attribute.

        Args:
            index: valid index for numpy ndarrays.
        Returns:
            Tensor wrapping the result of indexing into the data attribute.
        """
        return Tensor(self.data[index], req_grad=self.req_grad)

    @property
    def shape(self) -> tuple[int, ...]:
        """Returns shape of underlying data attribute.
        """
        return self.data.shape

    @property
    def size(self) -> int:
        """Returns size of underlying data attribute.
        """
        return self.data.size

    @staticmethod
    def _axes_difference(
        a: tuple[int, ...],
        b: tuple[int, ...]) -> tuple[int, ...]:
        """Returns indices for which a's axes are different from b's axes.
        Args:
            a, b: shapes of ndarrays.
        Returns:
            tuple containing the indices on which the two shapes differ.
        """
        return tuple(
            i
            for i, (u, v) in enumerate(zip(a, b))
            if u != v
        )

    @staticmethod
    def _collapse_axes(a: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
        """Use the reverse the broadcasting the effect of numpy ndarrays.

        Args:
            a: ndarray on which the reverse the broadcast.
            shape: shape that ndarray should take after reversing the broadcast.
        Returns:
            ndarray that is now shape of `shape` after collapsing the redundant axes
            using np.sum.
        """
        assert len(shape) <= a.ndim
        if len(shape) == 0: return np.sum(a)
        if a.shape == shape: return a

        if a.ndim != len(shape):
            missing = tuple(1 for _ in range(a.ndim - len(shape)))
            new_shape = tuple([*missing, *shape])
        else:
            new_shape = shape

        axes_diff = Tensor._axes_difference(a.shape, new_shape)
        r = np.sum(a, axis=axes_diff).reshape(shape)
        return r

    def backward(self):
        """Run a backward pass and store gradient in inputs .grad attribute.

        Once all the functions have finish executing we can run `backward` to compute
            the gradient of the function which respect to all of its inputs.

        Example:
            > a = t.Tensor(np.array([[1, 2]]))
            > b = t.Tensor(np.array([[3, 4]]))
            > c = a + b
            > c.backward()
            > a.grad
              [[1, 1]]
            > b.grad
              [[1, 1]]
        """
        end = -1
        for i, item in enumerate(_tape):
            if item.output is self:
                end = i + 1
                break

        if end == -1:
            s = ".backward() invoked on untraced Tensor"
            raise RuntimeError(s)

        if self.data.ndim == 0:
            self.grad = np.array(1, dtype=np.float64)
        else:
            s = ".backward() only works on scalar output tensors"
            raise RuntimeError(s)

        for item in reversed(_tape[:end]):
            inputs = item.inputs
            for i in inputs:
                if not i.req_grad: continue
                i.grad = np.zeros_like(i.data)

        for item in reversed(_tape[:end]):
            grads = item.backward(item.ctx, item.output.grad)
            for i, g in zip(item.inputs, grads):
                if not i.req_grad: continue
                i.grad += Tensor._collapse_axes(g, i.shape)

        _tape.clear()

    @staticmethod
    def _raise_typeerror(a, b, op_str) -> ty.NoReturn:
        """Raises TypeError when operation gets called with invalid inputs
        """
        a_cls = type(a).__name__
        b_cls = type(b).__name__
        s = f"unsupported operand types(s) for {op_str}: '{a_cls}' and '{b_cls}'"
        raise TypeError(s)

    @staticmethod
    def _check_and_generate(op: Callable, op_str: str) -> tuple[Callable, Callable]:
        """Generate the __op__ and __rop__ methods for a given operation

        Args:
            op: function that needed to be called when dunder method is called
            op_str: string representation of operation, useful for error messaging
        Returns:
            tuple which contain __op__ and __rop__ function which the dunder methods
            will be assigned to.
        """
        def lop(a: 'Tensor', b: 'int | float | Tensor') -> 'Tensor':
            if isinstance(b, Tensor):
                _b = b
            elif isinstance(b, (int, float)):
                _b = Tensor(np.full_like(a.data, b))
            else:
                Tensor._raise_typeerror(a, b, op_str)
            return op(a, _b)

        def rop(b: 'Tensor', a: int | float) -> 'Tensor':
            if isinstance(a, (int, float)):
                _a = Tensor(np.full_like(b.data, a))
            else:
                Tensor._raise_typeerror(a, b, op_str)
            return op(_a, b)

        return lop, rop

    @staticmethod
    def _add_F(ctx: _BContext, a: 'Tensor', b: 'Tensor') -> 'Tensor':
        return Tensor(a.data + b.data)

    @staticmethod
    def _add_B(ctx: _BContext, g: np.ndarray) -> tuple[optional_array, ...]:
        da = g if ctx.input_req_grads[0] else None
        db = g if ctx.input_req_grads[1] else None
        return (da, db)


    @staticmethod
    def _sub_F(ctx: _BContext, a: 'Tensor', b: 'Tensor') -> 'Tensor':
        return Tensor(a.data - b.data)

    @staticmethod
    def _sub_B(ctx: _BContext, g: np.ndarray) -> tuple[optional_array, ...]:
        da = +g if ctx.input_req_grads[0] else None
        db = -g if ctx.input_req_grads[1] else None
        return (da, db)


    @staticmethod
    def _mul_F(ctx: _BContext, a: 'Tensor', b: 'Tensor') -> 'Tensor':
        ctx.save_array(a.data)
        ctx.save_array(b.data)
        return Tensor(a.data * b.data)

    @staticmethod
    def _mul_B(ctx: _BContext, g: np.ndarray) -> tuple[optional_array, ...]:
        a, b, = ctx.saved_arrays
        da = g * b if ctx.input_req_grads[0] else None
        db = g * a if ctx.input_req_grads[1] else None
        return (da, db)


    @staticmethod
    def _div_F(ctx: _BContext, a: 'Tensor', b: 'Tensor') -> 'Tensor':
        _1b = 1 / b.data
        ctx.save_array(a.data)
        ctx.save_array(_1b)
        return Tensor(a.data * _1b)

    @staticmethod
    def _div_B(ctx: _BContext, g: np.ndarray) -> tuple[optional_array, ...]:
        a, _1b, = ctx.saved_arrays
        da = g * _1b if ctx.input_req_grads[0] else None
        db = g * (-a * _1b ** 2) if ctx.input_req_grads[1] else None
        return (da, db)


    @staticmethod
    def _matmul_F(ctx: _BContext, a: 'Tensor', b: 'Tensor') -> 'Tensor':
        ctx.save_array(a.data)
        ctx.save_array(b.data)
        return Tensor(a.data @ b.data)

    @staticmethod
    def _matmul_B(ctx: _BContext, g: np.ndarray) -> tuple[optional_array, ...]:
        a, b = ctx.saved_arrays
        da = g @ _T(b) if ctx.input_req_grads[0] else None
        db = _T(a) @ g if ctx.input_req_grads[1] else None
        return (da, db)


    @staticmethod
    def _neg_F(ctx: _BContext, a: 'Tensor') -> 'Tensor':
        return Tensor(-a.data)

    @staticmethod
    def _neg_B(ctx: _BContext, g: np.ndarray) -> tuple[optional_array, ...]:
        da = -g if ctx.input_req_grads[0] else None
        return (da, )


    @staticmethod
    def _T_F(ctx: _BContext, a: 'Tensor') -> 'Tensor':
        return Tensor(_T(a.data))

    @staticmethod
    def _T_B(ctx: _BContext, g: np.ndarray) -> tuple[optional_array, ...]:
        da = _T(g) if ctx.input_req_grads[0] else None
        return (da, )

    _add = partial(forward_all, _add_F, _add_B)
    _sub = partial(forward_all, _sub_F, _sub_B)
    _mul = partial(forward_all, _mul_F, _mul_B)
    _div = partial(forward_all, _div_F, _div_B)
    _matmul = partial(forward_all, _matmul_F, _matmul_B)
    _neg = partial(forward_all, _neg_F, _neg_B)
    _transpose = partial(forward_all, _T_F, _T_B)

    __add__, __radd__ = _check_and_generate(_add, '+')
    __sub__, __rsub__ = _check_and_generate(_sub, '-')
    __mul__, __rmul__ = _check_and_generate(_mul, '*')
    __truediv__, __rtruediv__ = _check_and_generate(_div, '/')

    def __neg__(self) -> 'Tensor':
        return Tensor._neg(self)

    @property
    def T(self) -> 'Tensor':
        return Tensor._transpose(self)

    def __matmul__(self, b: 'Tensor') -> 'Tensor':
        if not isinstance(b, Tensor):
            Tensor._raise_typeerror(self, b, '@')

        return Tensor._matmul(self, b)


def zeros(shape, req_grad=False) -> Tensor:
    """Create a zero tensor."""
    return Tensor(np.zeros(shape), req_grad=req_grad)

def zeros_like(t: Tensor, req_grad=False) -> Tensor:
    """Create a zero tensor of shape t.shape"""
    return Tensor(np.zeros_like(t.data), req_grad=req_grad)

def random(shape, req_grad=False):
    """Generate a tensor filled with random values."""
    return Tensor(_rng.random(shape), req_grad=req_grad)

def tape_clear():
    _tape.clear()

def tape_print():
    for item in _tape:
        print(item.backward.__qualname__)
        print(item.inputs)
        print(item.output)
        print()
