from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Tuple, Protocol, runtime_checkable


# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
    ----
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
    -------
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$

    """
    vals_plus = list(vals)
    vals_minus = list(vals)

    # Calculate f(x_0, ..., x_i + epsilon, ..., x_{n-1})
    vals_plus[arg] = vals_plus[arg] + epsilon
    f_plus = f(*vals_plus)

    # Calculate f(x_0, ..., x_i - epsilon, ..., x_{n-1})
    vals_minus[arg] = vals_minus[arg] - epsilon
    f_minus = f(*vals_minus)

    derivative = (f_plus - f_minus) / (2 * epsilon)
    return derivative


variable_count = 1


@runtime_checkable
class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        """Accumulate the derivative of the variable."""
        ...

    @property
    def unique_id(self) -> int:
        """Unique identifier for the variable."""
        ...

    def is_leaf(self) -> bool:
        """Is the variable a leaf node?"""
        ...

    def is_constant(self) -> bool:
        """Is the variable a constant?"""
        ...

    @property
    def parents(self) -> Iterable["Variable"]:
        """Parents of the variable in the computation graph."""
        ...

    def chain_rule(self, d_output: Any) -> Iterable[Tuple[Variable, Any]]:
        """Chain rule for backpropagation."""
        ...


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """Compute the topological order of the computation graph.

    Args:
    ----
        variable: The right-most variable

    Returns:
    -------
        Non-constant Variables in topological order starting from the right.

    """
    order = []
    seen = set()

    def visit(var: Variable) -> None:
        """Visit the variable and its parents."""
        if var.unique_id in seen or var.is_constant():
            return
        if not var.is_leaf():
            for m in var.parents:
                if not m.is_constant():
                    visit(m)
        seen.add(var.unique_id)
        order.insert(0, var)

    visit(variable)
    return order


def backpropagate(variable: Variable, deriv: Any) -> None:
    """Run backpropagation on the computation graph to compute derivatives for the leaf nodes.

    Args:
    ----
        variable: The right-most variable
        deriv: Its derivative that we want to propagate backward to the leaves.

    Returns:
    -------
        No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.

    """
    queue = topological_sort(variable)
    derivatives = {}
    derivatives[variable.unique_id] = deriv
    for var in queue:
        deriv = derivatives[var.unique_id]
        if var.is_leaf():
            var.accumulate_derivative(deriv)
        else:
            for v, d in var.chain_rule(deriv):
                if v.is_constant():
                    continue
                derivatives.setdefault(v.unique_id, 0.0)
                derivatives[v.unique_id] = derivatives[v.unique_id] + d


@dataclass
class Context:
    """Context class is used by `Function` to store information during the forward pass."""

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        """Store the given `values` if they need to be used during backpropagation."""
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        """Saved tensor."""
        return self.saved_values
