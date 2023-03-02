from __future__ import annotations

import math

import numpy as np


def count_graphs(n: int) -> float:
    edges = n * (n - 1) / 2
    return 2**edges


def binomial_coeff(n: int, k: int) -> float:
    return math.factorial(n) / (math.factorial(n - k))


def distance_from_origin(x: int, y: int) -> float:
    return np.sqrt(x**2 + y**2)


def generate_binary_strings(bit_count):
    binary_strings = []

    def genbin(n, bs=""):
        if len(bs) == n:
            binary_strings.append(bs)
        else:
            genbin(n, bs + "0")
            genbin(n, bs + "1")

    genbin(bit_count)
    return binary_strings


def cycle_list(l, loops=1):
    n = len(l)
    for _t in range(loops):
        l = [l[(i + 1) % n] for i in range(n)]
    return l


def generate_unique_combinations(L):
    combinations = generate_binary_strings(L)
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

    for i in np.arange(len(non_repeating)):
        non_repeating[i] = [int(s) for s in list(non_repeating[i])]
    return non_repeating