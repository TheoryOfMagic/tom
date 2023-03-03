from __future__ import annotations

import math
import typing as t

import numpy as np
import numpy.typing as npt

# Types

Int = np.int_ | int
Float = np.float_ | float

NDArrayInt = npt.NDArray[np.int_]
NDArrayFloat = npt.NDArray[np.float_]
NDArray = NDArrayInt | NDArrayFloat


def count_graphs(n: Int) -> Float:
    edges = n * (n - 1) / 2
    return 2**edges


def binomial_coeff(n: Int, k: Int) -> Float:
    return np.divide(math.factorial(n), (math.factorial(n - k)))


def distance_from_origin(x: Float, y: Float) -> Float:
    return np.sqrt(x**2 + y**2)


def generate_binary_strings(bit_count: Int) -> list[str]:
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
    n = len(loop)
    for _t in range(loops):
        loop = [loop[(i + 1) % n] for i in range(n)]
    return loop


def generate_unique_combinations(n: Int) -> list[list[Int]]:
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
