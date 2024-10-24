"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable


def mul(x: float, y: float) -> float:
    """Multiplies two numbers."""
    return x * y


def id(x: float) -> float:
    """Returns the input unchanged."""
    return x


def add(x: float, y: float) -> float:
    """Adds two numbers."""
    return x + y


def neg(x: float) -> float:
    """Negates a numbers."""
    return -x


def lt(x: float, y: float) -> float:
    """Checks if one number is less than another."""
    return 1.0 if x < y else 0.0


def eq(x: float, y: float) -> float:
    """Checks if two numbers are equal"""
    return 1.0 if x == y else 0.0


def max(x: float, y: float) -> float:
    """Returns the larger of two numbers."""
    return x if x > y else y


def is_close(x: float, y: float) -> float:
    """Checks if two numbers are close to each other."""
    return True if abs(x - y) < 1e-2 else False


def sigmoid(x: float) -> float:
    """Calculates the sigmoid function."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        return math.exp(x) / (1.0 + math.exp(x))


def relu(x: float) -> float:
    """Applies the ReLU activation function."""
    return x if x > 0 else 0.0


def log(x: float) -> float:
    """Calculates the natural logarithm."""
    if x <= 0:
        raise ValueError("x should be greater than 0")
    return math.log(x)


def exp(x: float) -> float:
    """Calculates the exponential function."""
    return math.exp(x)


def log_back(x: float, y: float) -> float:
    """Computes the derivative of log times a second arg."""
    return y * (1 / x)


def inv(x: float) -> float:
    """Calculates the reciprocal."""
    if x == 0:
        raise ValueError("x should not be 0")
    return 1 / x


def inv_back(x: float, y: float) -> float:
    """Computes the derivative of reciprocal times a second arg."""
    return -1 * (1 / (x**2)) * y


def relu_back(x: float, y: float) -> float:
    """Computes the derivative of ReLU times a second arg."""
    return (1 if x > 0 else 0) * y


# ## Task 0.3

# Small practice library of elementary higher-order functions.


# Implement the following core functions
def map(fn: Callable[[float], float], x: Iterable[float]) -> Iterable[float]:
    """Map func will apply the function to each element of the list

    Args:
    ----
        fn (Callable[[float], float]): A function that takes a float as input and returns a float.
        x (Iterable[float]): An iterable containing float values.

    Returns:
    -------
        Iterable[float]: A new iterable where the function `fn` has been applied to each element of `x`.

    """
    return [fn(x) for x in x]


def zipWith(
    fn: Callable[[float, float], float], x: Iterable[float], y: Iterable[float]
) -> Iterable[float]:
    """ZipWith func will apply the function to the elements of the list

    Args:
    ----
        fn (Callable[[float, float], float]): A function that takes two floats as input and returns a float.
        x (Iterable[float]): The first iterable of values.
        y (Iterable[float]): The second iterable of values.

    Returns:
    -------
        Iterable[float]: A new iterable where the function `fn` has been applied to pairs of elements from a and y.

    """
    result = []
    x_iter = iter(x)
    y_iter = iter(y)

    while True:
        try:
            x_val = next(x_iter)
            y_val = next(y_iter)
            result.append(fn(x_val, y_val))
        except StopIteration:
            break
    return result


def reduce(
    fn: Callable[[float, float], float], iterable: Iterable[float], initial: float = 0.0
) -> float:
    """Reduces an iterable to a single value using the function.

    Args:
    ----
        fn (Callable[[float, float], float]): A function that takes two floats and returns a single float.
        iterable (Iterable[float]): The iterable containing float values to be reduced.
        initial (float, optional): The initial value to start the reduction, which is defaulted to 0.0.

    Returns:
    -------
        float: A single value that is the result of repeatedly applying the function `fn` to the elements of `iterable`.

    """
    it = iter(iterable)
    try:
        value = next(it)
    except StopIteration:
        return initial
    for element in it:
        value = fn(value, element)
    return value


# Use these to implement
# - negList : negate a list
def negList(lst: Iterable[float]) -> Iterable[float]:
    """Negate all elements in a list."""
    return map(lambda x: -x, lst)


# - addLists : add two lists together
def addLists(lst1: Iterable[float], lst2: Iterable[float]) -> Iterable[float]:
    """Add corresponding elements from two lists."""
    return zipWith(add, lst1, lst2)


# - sum: sum lists
def sum(lst: Iterable[float]) -> float:
    """Sum all elements in a list."""
    return reduce(add, lst)


# - prod: take the product of lists
def prod(lst: Iterable[float]) -> float:
    """Product of all elements in a list."""
    return reduce(mul, lst)
