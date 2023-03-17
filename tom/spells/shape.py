from __future__ import annotations

import math
import typing as t

import matplotlib.pyplot as plt_
import numpy as np

from tom.spells.geometry import RadialSpellGeometryParams
from tom.spells.geometry import SpellGeometries
from tom.spells.geometry import SpellGeometry
from tom.util.math import NDArray
from tom.util.math import NDArrayFloat
from tom.util.math import NDArrayInt


def draw_centre_circle(
    P: NDArray,
    Q: NDArray,
    thetas: str | None = None,
) -> tuple[NDArray, NDArray, float, float]:
    x1 = P[0]
    y1 = P[1]
    x2 = Q[0]
    y2 = Q[1]
    a = (x1 + x2) / 2
    b = (y1 + y2) / 2
    r = np.sqrt((a - x1) ** 2 + (b - y1) ** 2)

    theta: NDArray
    if thetas == "Full":
        theta = np.linspace(0, 2 * np.pi, 150)
    else:
        theta0 = math.atan2(y1 - b, x1 - a)
        theta1 = math.atan2(y2 - b, x2 - a)
        if y2 < y1:
            theta0, theta1 = theta1 + np.pi, theta0 + np.pi
        theta = np.linspace(theta0, theta1, 150)

    X2: NDArray = r * np.cos(theta) + a
    Y2: NDArray = r * np.sin(theta) + b

    return (X2, Y2, a, b)


def draw_non_centre_circle(
    P: NDArray,
    Q: NDArray,
    b: float,
    radius: float = 0.0,
    thetas: str | None = None,
) -> tuple[NDArray, NDArray, float, float]:
    x1 = P[0]
    y1 = P[1]
    x2 = Q[0]
    y2 = Q[1]

    b2 = -b

    delta = x1**2 - x2**2 + y1**2 - y2**2
    a = (delta - 2 * (y1 - y2) * b) / (2 * (x1 - x2))
    a2 = (delta - 2 * (y1 - y2) * b2) / (2 * (x1 - x2))

    r = np.sqrt((x1 - a) ** 2 + (y1 - b) ** 2)
    r2 = np.sqrt((x1 - a2) ** 2 + (y1 - b2) ** 2)
    if r2 <= r:
        a = a2
        b = b2
        r = r2

    theta: NDArray
    if thetas == "Full":
        theta = np.linspace(0, 2 * np.pi, 150)
    else:
        theta0 = math.atan2(y1 - b, x1 - a)
        theta1 = math.atan2(y2 - b, x2 - a)

        theta02 = theta0
        theta12 = theta1

        while theta1 < theta0:
            theta0 -= 2 * np.pi

        while theta02 < theta12:
            theta12 -= 2 * np.pi

        arc1 = r * (theta1 - theta0)
        arc2 = r * (theta02 - theta12)

        if arc1 < arc2 or np.sqrt(b**2) < 1:
            theta = np.linspace(theta1, theta0, 150)
        else:
            theta = np.linspace(theta02, theta12, 150)

    X: NDArrayFloat = r * np.cos(theta) + a
    Y: NDArrayFloat = r * np.sin(theta) + b

    return (X, Y, a, b)


def decode(
    feature_array: list[list[int]],
    radius: float = 1.0,
    start_angle: float | None = None,
    color: str = "k",
    label: str | None = None,
    base: SpellGeometries = SpellGeometries.polygon,
    plt: t.Any = plt_,
):
    for k, in_array in enumerate(feature_array, 1):
        params = RadialSpellGeometryParams(
            node_count=len(in_array),
            start_angle=np.pi / len(in_array) if start_angle is None else start_angle,
            radius=radius,
        )

        geometry = SpellGeometry.build_with(base, params)

        x, y, n = geometry.x, geometry.y, params.n

        labelled = False
        for i in range(n):
            if in_array[i] == 1:
                if labelled == False:
                    plt.plot(
                        [x[i], x[(i + k) % n]],
                        [y[i], y[(i + k) % n]],
                        "-",
                        color=color,
                        label=label,
                    )
                    labelled = True
                else:
                    plt.plot(
                        [x[i], x[(i + k) % n]], [y[i], y[(i + k) % n]], "-", color=color
                    )
            elif in_array[i] == 2:
                offset = 0.01
                plt.plot(
                    [x[i], x[(i + k) % n]],
                    [y[i], y[(i + k) % n]],
                    "-",
                    color=color,
                    label=label,
                )
                plt.plot(
                    [x[i] + offset, x[(i + k) % n] + offset],
                    [y[i] + offset, y[(i + k) % n] + offset],
                    "-",
                    color=color,
                )
            else:
                plt.plot(
                    [x[i], x[(i + k) % n]],
                    [y[i], y[(i + k) % n]],
                    ":",
                    linewidth=0.5,
                    color=color,
                )
        plt.scatter(x, y, s=70, facecolors="none", edgecolors="k")
        plt.axis("scaled")
        plt.axis("off")
    # plt.show()


def decode_shape_circular(
    in_array: NDArrayInt | list[int],
    k: int = 1,
    radius: int = 1,
    start_angle: float | None = None,
    label: str | None = None,
    color: str = "k",
    centered: bool = True,
    s: int = 0,
    base: str = "Polygon",
    plt: t.Any = plt_,
):
    if np.sqrt(s**2) < 1 and "Quadratic" in base:
        raise Exception("Exception: Quadratic shapes requires s >= 1")
    n = len(in_array)
    if start_angle == None:
        start_angle = np.pi / n

    x: NDArray = np.array([])
    y: NDArray = np.array([])

    if base == "Polygon":
        small_angle = np.fromiter(
            (start_angle + i * 2 * np.pi / n for i in np.arange(1, n + 1)),
            np.float_,
        )
        x, y = (radius * np.sin(small_angle), radius * np.cos(small_angle))
    elif base == "Line":
        x = np.arange(0, n)
        y = np.zeros((1, n))[0]
    elif base == "Quadratic":
        x = np.arange(-math.floor(n / 2), math.ceil(n / 2))
        y = np.array(x_**2 for x_ in x)
    elif base == "Quadratic2":
        x = np.array([0])
        while len(x) < n:
            x = np.append(
                x,
                (-x[-1] + 1) if -x[-1] in x else (-x[-1]),
            )
        y = np.array(x_**2 for x_ in x)

    elif base == "SemiCircular":
        theta0 = 0
        theta1 = -np.pi
        theta = np.linspace(theta0, theta1, n)
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
    elif base == "QuarterCircular":
        theta0 = 0
        theta1 = -np.pi / 2
        theta = np.linspace(theta0, theta1, n)
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
    elif base == "CubicFunction":
        x = np.arange(-math.floor(n / 2), math.ceil(n / 2))
        y = np.fromiter((0.1 * x_**3 + -0.75 * x_ for x_ in x), np.float_)
    labelled = label == None

    for i in range(n):
        P = np.array([x[i], y[i]])
        Q = np.array([x[(i + k) % n], y[(i + k) % n]])

        X, Y, _, _ = (
            draw_centre_circle(P, Q)
            if centered
            else draw_non_centre_circle(P, Q, radius=radius, b=s)
        )

        # plt.plot(a,b,"x",color = color)
        if in_array[i] == 1:
            if labelled == False:
                plt.plot(X, Y, label=label, color=color)
                labelled = True
            else:
                plt.plot(X, Y, color=color)

        else:
            plt.plot(X, Y, ":", color=color, linewidth=0.3)
    plt.scatter(x, y, s=70, facecolors="none", edgecolors="k")

    plt.axis("scaled")
    plt.axis("off")
