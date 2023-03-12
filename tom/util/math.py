"""Utility maths library."""

from __future__ import annotations

import math
import typing as t
from enum import Enum
from enum import auto

import numpy as np
import numpy.typing as npt

# Types

Int = np.int_ | int
Float = np.float_ | float

NDArrayInt = npt.NDArray[np.int_]
NDArrayFloat = npt.NDArray[np.float_]
NDArray = NDArrayInt | NDArrayFloat


def count_graphs(n: Int) -> Float:
    """Count subgraphs contained within a set of nodes.

    Parameters
    ----------
    n : Int
        The total number of nodes in the graph.

    """
    edges = n * (n - 1) / 2
    return 2**edges


def generate_binary_strings(bit_count: Int) -> list[str]:
    """Generate all valid binary strings up to length bit_count.

    Parameters
    ----------
    bit_count : Int
        The binary string bit count.

    """
    binary_strings: list[str] = []

    def genbin(n: Int, bs: str = "") -> None:
        if len(bs) == n:
            binary_strings.append(bs)
        else:
            genbin(n, bs + "0")
            genbin(n, bs + "1")

    genbin(bit_count)
    return binary_strings


def cycle_list(loop: list[t.Any], loops: Int = 1):
    """Rotate an array by the given number of loops.

    Parameters
    ----------
    loop : list[any]
        The array to rotate.
    loops : Int
        The number of cycles to loop.

    """
    return loop[loops:] + loop[:loops]


def generate_unique_combinations(n: Int) -> list[list[Int]]:
    """Generate all binary strings that are unique respective to rotational symmetry.

    This method attempts to identify all rotational equivalence classes for byte strings
    up to a given length. In otherwords, all binary strings that are rotations of
    each other are considered equivalent and this method attempts to classify those
    varying groups. We then identify each class by its minimal element and return a list
    of those elements.

    Parameters
    ----------
    n : Int
        The binary string bit count.

    TODO
    ----
     - This method can *definitely* be optimized. Let's optimize it :)

    """
    combinations = generate_binary_strings(n)
    non_repeating = [combinations[0]]
    for i in range(len(combinations)):
        ref = list(combinations[i])
        N = len(ref)
        test = 0
        for j in range(len(non_repeating)):
            for n in range(N):
                if cycle_list(list(non_repeating[j]), loops=n + 1) == ref:
                    test += 1

        if test == 0:
            non_repeating.append(combinations[i])

    int_non_repeating: list[list[Int]] = [[int(s) for s in nr] for nr in non_repeating]
    return int_non_repeating


class ShapeBase(Enum):
    polygon = auto()
    line = auto()
    quadratic = auto()
    quadratic2 = auto()
    semi_circular = auto()
    quarter_circular = auto()
    cubic_function = auto()


def _decode_path(
    in_array: NDArrayInt | list[int],
    k: int,
    start_angle: float | None = None,
    base: ShapeBase = ShapeBase.polygon,
    radius: int | None = None,  # circular path if non-null
    s: int = 0,  # only relevant for circular paths
    centered: bool = True,  # only relevant for circular paths
) -> NDArrayFloat:
    path: NDArrayFloat = np.array([])

    n = len(in_array)

    if start_angle == None:
        start_angle = np.pi / n

    x: NDArray = np.array([])
    y: NDArray = np.array([])

    if base == ShapeBase.polygon:
        small_angle = np.fromiter(
            (start_angle + i * 2 * np.pi / n for i in np.arange(1, n + 1)),
            np.float_,
        )

        x = radius * np.sin(small_angle)
        y = radius * np.cos(small_angle)

    elif base == ShapeBase.line:
        x = np.arange(0, n)
        y = np.zeros(n)

    elif base == ShapeBase.quadratic:
        x = np.arange(-math.floor(n / 2), math.ceil(n / 2))
        y = np.array(x_**2 for x_ in x)

    elif base == ShapeBase.quadratic2:
        x = np.array([0])
        while len(x) < n:
            x = np.append(
                x,
                (-x[-1] + 1) if -x[-1] in x else (-x[-1]),
            )
        y = np.array(x_**2 for x_ in x)

    elif base == ShapeBase.semi_circular:
        theta0 = 0
        theta1 = -np.pi
        theta = np.linspace(theta0, theta1, n)
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)

    elif base == ShapeBase.quarter_circular:
        theta0 = 0
        theta1 = -np.pi / 2
        theta = np.linspace(theta0, theta1, n)
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)

    elif base == ShapeBase.cubic_function:
        x = np.arange(-math.floor(n / 2), math.ceil(n / 2))
        y = np.array(0.1 * x_**3 + -0.75 * x_ for x_ in x)

    for i in range(n):
        P = np.array([x[i], y[i]])
        Q = np.array([x[(i + k) % n], y[(i + k) % n]])

    print(P, Q)
    return path
