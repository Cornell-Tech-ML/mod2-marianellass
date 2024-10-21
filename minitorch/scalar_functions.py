from __future__ import annotations

from typing import TYPE_CHECKING
import math
import minitorch

from . import operators
from .autodiff import Context


if TYPE_CHECKING:
    from typing import Tuple

    from .scalar import Scalar, ScalarLike


def wrap_tuple(x: float | Tuple[float, ...]) -> Tuple[float, ...]:
    """Turn a possible value into a tuple"""
    if isinstance(x, tuple):
        return x
    return (x,)


class ScalarFunction:
    """A wrapper for a mathematical function that processes and produces
    Scalar variables.

    This is a static class and is never instantiated. We use `class`
    here to group together the `forward` and `backward` code.
    """

    @classmethod
    def _backward(cls, ctx: Context, d_out: float) -> Tuple[float, ...]:
        return wrap_tuple(cls.backward(ctx, d_out))  # type: ignore

    @classmethod
    def _forward(cls, ctx: Context, *inps: float) -> float:
        return cls.forward(ctx, *inps)  # type: ignore

    @classmethod
    def apply(cls, *vals: ScalarLike) -> Scalar:
        """Apply the function to a set of Scalar values."""
        raw_vals = []
        scalars = []
        for v in vals:
            if isinstance(v, minitorch.scalar.Scalar):
                scalars.append(v)
                raw_vals.append(v.data)
            else:
                scalars.append(minitorch.scalar.Scalar(v))
                raw_vals.append(v)

        # Create the context.
        ctx = Context(False)

        # Call forward with the variables.
        c = cls._forward(ctx, *raw_vals)
        assert isinstance(c, float), "Expected return type float got %s" % (type(c))

        # Create a new variable from the result with a new history.
        back = minitorch.scalar.ScalarHistory(cls, ctx, scalars)
        return minitorch.scalar.Scalar(c, back)


# Examples
class Add(ScalarFunction):
    """Addition function $f(x, y) = x + y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Add two numbers."""
        return a + b

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Add two numbers."""
        return d_output, d_output


class Log(ScalarFunction):
    """Log function $f(x) = log(x)$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Apply the natural logarithm to a number."""
        ctx.save_for_backward(a)
        return operators.log(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Apply the natural logarithm to a number."""
        (a,) = ctx.saved_values
        return operators.log_back(a, d_output)


class Mul(ScalarFunction):
    """Multiplication function f(x, y) = x * y"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Multiply two numbers."""
        ctx.save_for_backward(a, b)
        return a * b

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Retrieve saved values a and b."""
        a, b = ctx.saved_values
        return b * d_output, a * d_output


class Inv(ScalarFunction):
    """Inverse function f(x) = 1 / x"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Apply the inverse to a number."""
        ctx.save_for_backward(a)
        return 1 / a

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Apply the inverse to a number."""
        (a,) = ctx.saved_values
        return -1 / (a**2) * d_output


class Neg(ScalarFunction):
    """Negation function f(x) = -x"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Apply the negation to a number."""
        return float(-a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Apply the negation to a number."""
        return -d_output


class Sigmoid(ScalarFunction):
    """Sigmoid function f(x) = 1 / (1 + exp(-x))"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Apply the sigmoid function to a number."""
        sig = 1 / (1 + math.exp(-a))
        ctx.save_for_backward(sig)
        return sig

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Apply the sigmoid function to a number."""
        (sig,) = ctx.saved_values
        return sig * (1 - sig) * d_output


class ReLU(ScalarFunction):
    """ReLU function f(x) = max(0, x)"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Apply the ReLU function to a number."""
        ctx.save_for_backward(a)
        return float(max(0, a))

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Apply the ReLU function to a number."""
        (a,) = ctx.saved_values
        return d_output if a > 0 else 0.0


class Exp(ScalarFunction):
    """Exponential function f(x) = exp(x)"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Apply the exponential function to a number."""
        result = math.exp(a)
        ctx.save_for_backward(result)
        return result

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Apply the exponential function to a number."""
        (result,) = ctx.saved_values
        return result * d_output


class Lt(ScalarFunction):
    """Less than function f(x, y) = x < y"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Apply the less than function to two numbers."""
        return 1.0 if a < b else 0.0

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Apply the less than function to two numbers."""
        return 0.0, 0.0


class Eq(ScalarFunction):
    """Equality function f(x, y) = x == y"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Apply the equality function to two numbers returning either 0 or 1."""
        return 1.0 if a == b else 0.0

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Derivative of a constant is 0."""
        return 0.0, 0.0


class is_close(ScalarFunction):
    """Equality function f(x, y) = x == y"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float, tol: float) -> float:
        """Apply the equality function to two numbers returning either 0 or 1."""
        return 1.0 if abs(a - b) < tol else 0.0

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Derivative of a constant is 0."""
        return 0.0, 0.0, 0.0
