"""Implementation of the autodifferentiation Functions for Tensor."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, Optional

import numpy as np

import minitorch

from . import operators
from .autodiff import Context
from .tensor_ops import SimpleBackend, TensorBackend

if TYPE_CHECKING:
    from typing import Any, List, Tuple

    from .tensor import Tensor
    from .tensor_data import UserIndex, UserShape


def wrap_tuple(x: Any) -> tuple:  # type: ignore
    """Turn a possible value into a tuple"""
    if isinstance(x, tuple):
        return x
    return (x,)


# Constructors
class Function:
    @classmethod
    def _backward(cls, ctx: Context, grad_out: Tensor) -> Tuple[Tensor, ...]:
        return wrap_tuple(cls.backward(ctx, grad_out))  # type: ignore

    @classmethod
    def _forward(cls, ctx: Context, *inps: Tensor) -> Tensor:
        return cls.forward(ctx, *inps)  # type: ignore

    @classmethod
    def apply(cls, *vals: Any) -> Tensor:
        """Call the forward function and track history"""
        from .tensor import Tensor  # Local import to avoid circular dependency

        raw_vals = []
        need_grad = False
        for v in vals:
            if isinstance(v, Tensor):
                if v.requires_grad():
                    need_grad = True
                raw_vals.append(v.detach())
            else:
                raw_vals.append(v)

        # Create the context.
        ctx = Context(not need_grad)

        # Call forward with the variables.
        c = cls._forward(ctx, *raw_vals)

        # Create a new variable from the result with a new history.
        back = None
        if need_grad:
            back = minitorch.History(cls, ctx, vals)
        return minitorch.Tensor(c._tensor, back, backend=c.backend)


class Neg(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        return t1.f.neg_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        return grad_output.f.neg_map(grad_output)


class Inv(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        ctx.save_for_backward(t1)
        return t1.f.inv_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        (t1,) = ctx.saved_values
        return grad_output.f.inv_back_zip(t1, grad_output)


class Add(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        return t1.f.add_zip(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        return grad_output, grad_output


class All(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, dim: Optional[Tensor] = None) -> Tensor:
        """Return 1 if all are true"""
        if dim is not None:
            return a.f.mul_reduce(a, int(dim.item()))
        else:
            return a.f.mul_reduce(a.contiguous().view(int(operators.prod(a.shape))), 0)


class Mul(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Matrix Multiply Forward"""
        return t1.f.mul_zip(t1, t2)  # Custom multiplication logic here

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Matrix Multiply backward"""
        return grad_output, grad_output  # Adjust as necessary


class Sum(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, dim: Optional[int] = None) -> Tensor:
        ctx.save_for_backward(a.shape, dim)
        if dim is None:
            # Flatten the tensor and sum over dimension 0
            flattened = a.contiguous().view(-1)
            result = a.f.sum_reduce(flattened, dim=0)
            # result is a scalar tensor with shape ()
            # To make it indexable with [0], reshape it to (1,)
            result = result.view(1)
        else:
            result = a.f.sum_reduce(a, dim)
        return result

    def backward(self, ctx: Context, grad_output: Tensor) -> Tuple[Tensor]:
        from .tensor import Tensor
        from .tensor_data import (
            TensorData,
        )

        a_shape, dim = ctx.saved_values
        if dim is None:
            # grad_output is a tensor of shape (1,)
            grad_output_value = grad_output[0]
            # Create a tensor of the same shape as 'a', filled with 'grad_output_value'
            total_elements = int(operators.prod(a_shape))
            grad_input_data = [grad_output_value] * total_elements
            grad_input_data = TensorData(grad_input_data, a_shape)
            grad_input = Tensor(grad_input_data)
        else:
            # Reshape grad_output to have singleton dimension at 'dim'
            grad_output_shape = [1 if i == dim else s for i, s in enumerate(a_shape)]
            grad_output = grad_output.view(*grad_output_shape)
            # Use broadcasting to match the shape of 'a'
            grad_input = grad_output + zeros(a_shape, backend=grad_output.backend)
        return (grad_input,)


class Sigmoid(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor) -> Tensor:
        """Sigmoid of tensor"""
        return a.f.sigmoid_map(a)


class Relu(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor) -> Tensor:
        """Relu of tensor"""
        return a.f.relu_map(a)


class Log(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor) -> Tensor:
        """Log of tensor"""
        return a.f.log_map(a)


class Exp(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor) -> Tensor:
        """Exp of tensor"""
        return a.f.exp_map(a)


class LT(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor) -> Tensor:
        """Less than"""
        return a.f.lt_zip(a, b)


class EQ(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor) -> Tensor:
        """Equal"""
        return a.f.eq_zip(a, b)


class IsClose(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor, tol: float) -> Tensor:
        ctx.save_for_backward(tol)
        return a.f.is_close_zip(a, b)


class permute(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, dims: Tuple[int]) -> Tensor:
        """Permute the dimensions of a tensor according to dims."""
        ctx.save_for_backward(dims)
        return a.permute(*dims)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, None]:
        (dims,) = ctx.saved_values
        # Compute the inverse permutation
        inv_dims = [0] * len(dims)
        for i, dim in enumerate(dims):
            inv_dims[dim] = i
        # Permute the gradient back to the original dimensions
        grad_input = grad_output.permute(*inv_dims)
        return grad_input, None  # No gradient w.r.t. dims


class View(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, shape: Tensor) -> Tensor:
        ctx.save_for_backward(a.shape)
        assert a._tensor.is_contiguous(), "Must be contiguous to view"
        shape2 = [int(shape[i]) for i in range(shape.size)]
        return minitorch.Tensor.make(
            a._tensor._storage, tuple(shape2), backend=a.backend
        )

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Matrix Multiply backward (module 3)"""
        (original,) = ctx.saved_values
        return (
            minitorch.Tensor.make(
                grad_output._tensor._storage, original, backend=grad_output.backend
            ),
            0.0,
        )


class Copy(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor) -> Tensor:
        """Id function makes contiguous"""
        return a.f.id_map(a)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Undo"""
        return grad_output


class MatMul(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Matrix Multiply Forward (module 3)"""
        ctx.save_for_backward(t1, t2)
        return t1.f.matrix_multiply(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Matrix Multiply backward (module 3)"""
        t1, t2 = ctx.saved_values

        def transpose(a: Tensor) -> Tensor:
            order = list(range(a.dims))
            order[-2], order[-1] = order[-1], order[-2]
            return a._new(a._tensor.permute(*order))

        return (
            grad_output.f.matrix_multiply(grad_output, transpose(t2)),
            grad_output.f.matrix_multiply(transpose(t1), grad_output),
        )


# Helpers for Constructing tensors
def zeros(shape: UserShape, backend: TensorBackend = SimpleBackend) -> Tensor:
    """Produce a zero tensor of size `shape`.

    Args:
    ----
        shape : shape of tensor
        backend : tensor backend

    Returns:
    -------
        new tensor

    """
    return minitorch.Tensor.make(
        [0.0] * int(operators.prod(shape)), shape, backend=backend
    )


def rand(
    shape: UserShape,
    backend: TensorBackend = SimpleBackend,
    requires_grad: bool = False,
) -> Tensor:
    """Produce a random tensor of size `shape`.

    Args:
    ----
        shape : shape of tensor
        backend : tensor backend
        requires_grad : turn on autodifferentiation

    Returns:
    -------
        :class:`Tensor` : new tensor

    """
    vals = [random.random() for _ in range(int(operators.prod(shape)))]
    tensor = minitorch.Tensor.make(vals, shape, backend=backend)
    tensor.requires_grad_(requires_grad)
    return tensor


def _tensor(
    ls: Any,
    shape: UserShape,
    backend: TensorBackend = SimpleBackend,
    requires_grad: bool = False,
) -> Tensor:
    """Produce a tensor with data ls and shape `shape`.

    Args:
    ----
        ls: data for tensor
        shape: shape of tensor
        backend: tensor backend
        requires_grad: turn on autodifferentiation

    Returns:
    -------
        new tensor

    """
    tensor = minitorch.Tensor.make(ls, shape, backend=backend)
    tensor.requires_grad_(requires_grad)
    return tensor


def tensor(
    ls: Any, backend: TensorBackend = SimpleBackend, requires_grad: bool = False
) -> Tensor:
    """Produce a tensor with data and shape from ls

    Args:
    ----
        ls: data for tensor
        backend : tensor backend
        requires_grad : turn on autodifferentiation

    Returns:
    -------
        :class:`Tensor` : new tensor

    """

    def shape(ls: Any) -> List[int]:
        if isinstance(ls, (list, tuple)):
            return [len(ls)] + shape(ls[0])
        else:
            return []

    def flatten(ls: Any) -> List[float]:
        if isinstance(ls, (list, tuple)):
            return [y for x in ls for y in flatten(x)]
        else:
            return [ls]

    cur = flatten(ls)
    shape2 = shape(ls)
    return _tensor(cur, tuple(shape2), backend=backend, requires_grad=requires_grad)


# Gradient check for tensors


def grad_central_difference(
    f: Any, *vals: Tensor, arg: int = 0, epsilon: float = 1e-6, ind: UserIndex
) -> float:
    x = vals[arg]
    up = zeros(x.shape)
    up[ind] = epsilon
    vals1 = [x if j != arg else x + up for j, x in enumerate(vals)]
    vals2 = [x if j != arg else x - up for j, x in enumerate(vals)]
    delta: Tensor = f(*vals1).sum() - f(*vals2).sum()

    return delta[0] / (2.0 * epsilon)


def grad_check(f: Any, *vals: Tensor) -> None:
    """Check whether autodiff matches central difference."""
    for x in vals:
        x.requires_grad_(True)
        x.zero_grad_()
    random.seed(10)
    out = f(*vals)
    out.sum().backward()
    err_msg = """

Gradient check error for function %s.

Input %s

Received derivative %f for argument %d and index %s,
but was expecting derivative %f from central difference.

"""

    for i, x in enumerate(vals):
        ind = x._tensor.sample()
        check = grad_central_difference(f, *vals, arg=i, ind=ind)
        assert x.grad is not None
        np.testing.assert_allclose(
            x.grad[ind],
            check,
            1e-2,
            1e-2,
            err_msg=err_msg % (f, vals, x.grad[ind], i, ind, check),
        )
