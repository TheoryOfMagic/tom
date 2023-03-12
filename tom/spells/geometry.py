from __future__ import annotations

import math
import typing as t
from abc import abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from tom.util.math import Float
from tom.util.math import Int
from tom.util.math import NDArray

SpellGeometryElement = tuple[Int | Float]


class SpellGeometry(Iterator[SpellGeometryElement]):
    """Geometry base class for a generic spell.

    This class stores a list of verticies and can be used idiomatically like a typical
    collection to grab the vertices at individual points
    """

    x: NDArray
    y: NDArray

    def __init__(self, x: NDArray, y: NDArray) -> None:
        self.x = x
        self.y = y

    def __iter__(self) -> Iterator[SpellGeometryElement]:
        # TODO: Implement
        raise NotImplementedError

    def __next__(self) -> SpellGeometryElement:
        # TODO: Implement
        raise NotImplementedError

    # Factory methods
    @staticmethod
    def build_with(strategy: SpellGeometryBuildStrategy[P], params: P) -> SpellGeometry:
        return strategy.build(params)


# Geometry Strategy Params

P = t.TypeVar("P", bound="SpellGeometryParams")


@dataclass(frozen=True)
class SpellGeometryParams:
    node_count: int = 1

    @property
    def n(self) -> int:
        return self.node_count


@dataclass(frozen=True)
class RadialSpellGeometryParams(SpellGeometryParams):
    start_angle: float | None = None
    radius: float = 1.0

    @property
    def t0(self) -> float | None:
        return self.start_angle

    @property
    def r(self) -> float:
        return self.radius


# Geometry Strategies


class SpellGeometryBuildStrategy(t.Generic[P]):
    def __init_subclass__(cls) -> None:
        return super().__init_subclass__()

    @abstractmethod
    @classmethod
    def build(cls, params: P) -> SpellGeometry:
        raise NotImplementedError


class LineSpell(SpellGeometryBuildStrategy[SpellGeometryParams]):
    @classmethod
    def build(cls, params: SpellGeometryParams) -> SpellGeometry:
        return SpellGeometry(
            x=np.arange(0, params.n),
            y=np.zeros(params.n),
        )


class PolygonSpell(SpellGeometryBuildStrategy[RadialSpellGeometryParams]):
    @classmethod
    def build(cls, params: RadialSpellGeometryParams) -> SpellGeometry:
        start_angle = np.pi / params.n if params.t0 is None else params.t0

        small_angle = np.fromiter(
            (
                start_angle + i * 2 * np.pi / params.n
                for i in np.arange(1, params.n + 1)
            ),
            np.float_,
        )

        return SpellGeometry(
            x=params.r * np.sin(small_angle),
            y=params.r * np.cos(small_angle),
        )


class QuadraticSpell(SpellGeometryBuildStrategy[SpellGeometryParams]):
    @classmethod
    def build_from_x(cls, x: NDArray) -> SpellGeometry:
        return SpellGeometry(x=x, y=np.array(x_**2 for x_ in x))

    @classmethod
    def build(cls, params: SpellGeometryParams) -> SpellGeometry:
        return cls.build_from_x(
            x=np.arange(-math.floor(params.n / 2), math.ceil(params.n / 2)),
        )


class QuadraticCenteredSpell(QuadraticSpell):
    @classmethod
    def build(cls, params: SpellGeometryParams) -> SpellGeometry:
        _x = np.array([0])
        while len(_x) < params.n:
            if -_x[-1] in _x:
                _x = np.append(_x, (-_x[-1] + 1) if -_x[-1] in _x else (-_x[-1]))

        return cls.build_from_x(x=_x)


class CubicSpell(SpellGeometryBuildStrategy[SpellGeometryParams]):
    @classmethod
    def build_from_x(cls, x: NDArray) -> SpellGeometry:
        return SpellGeometry(x=x, y=np.array(0.1 * x_**3 + -0.75 * x_ for x_ in x))

    @classmethod
    def build(cls, params: SpellGeometryParams) -> SpellGeometry:
        return cls.build_from_x(
            x=np.arange(-math.floor(params.n / 2), math.ceil(params.n / 2)),
        )


class SemiCircularSpell(SpellGeometryBuildStrategy[RadialSpellGeometryParams]):
    @classmethod
    def build(cls, params: RadialSpellGeometryParams) -> SpellGeometry:
        theta0 = 0
        theta1 = -np.pi
        theta = np.linspace(theta0, theta1, params.n)

        return SpellGeometry(
            x=params.r * np.cos(theta),
            y=params.r * np.sin(theta),
        )


class QuarterCircularSpell(SpellGeometryBuildStrategy[RadialSpellGeometryParams]):
    @classmethod
    def build(cls, params: RadialSpellGeometryParams) -> SpellGeometry:
        theta0 = 0
        theta1 = -np.pi / 2
        theta = np.linspace(theta0, theta1, params.n)

        return SpellGeometry(
            x=params.r * np.cos(theta),
            y=params.r * np.sin(theta),
        )
