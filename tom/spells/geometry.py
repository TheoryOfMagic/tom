from __future__ import annotations

import math
import typing as t
from abc import ABC
from abc import abstractmethod
from collections.abc import Iterator
from enum import Enum
from enum import auto

import numpy as np

from tom.spells.types import SpellGeometryParams
from tom.util.math import Float
from tom.util.math import Int
from tom.util.math import NDArrayFloat
from tom.util.math import NDArrayInt

SpellGeometryAxis = NDArrayInt | NDArrayFloat
SpellGeometryNode = Int | Float
SpellGeometryElement = tuple[SpellGeometryNode, SpellGeometryNode]


class SpellGeometries(Enum):
    line = auto()
    polygon = auto()
    quadratic = auto()
    quadratic_centered = auto()
    cubic = auto()
    semi_circle = auto()
    quarter_circle = auto()


class SpellGeometry(Iterator[SpellGeometryElement]):
    """Geometry base class for a generic spell.

    This class stores a list of verticies and can be used idiomatically like a typical
    collection to grab the vertices at individual points
    """

    x: SpellGeometryAxis
    y: SpellGeometryAxis

    def __init__(self, x: SpellGeometryAxis, y: SpellGeometryAxis) -> None:
        self.x = x
        self.y = y

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, index: int) -> SpellGeometryElement:
        return t.cast(SpellGeometryElement, (self.x[index], self.y[index]))

    def __iter__(self) -> Iterator[SpellGeometryElement]:
        for element in zip(self.x, self.y, strict=True):
            yield t.cast(SpellGeometryElement, element)

    def __next__(self) -> SpellGeometryElement:
        # TODO: Implement
        raise NotImplementedError

    # Factory methods
    @staticmethod
    def build_with(
        strategy: SpellGeometries | type[SpellGeometryBuildStrategy],
        params: SpellGeometryParams,
    ) -> SpellGeometry:
        if isinstance(strategy, SpellGeometries):
            strategy = _spell_geometry_strategies_registry[strategy]
        return strategy.build(params)


# Geometry Strategies

_spell_geometry_strategies_registry: dict[
    SpellGeometries, type[SpellGeometryBuildStrategy]
] = {}


class SpellGeometryBuildStrategy(ABC):
    geometry: SpellGeometries | None = None

    def __init_subclass__(cls) -> None:
        if cls == SpellGeometryBuildStrategy:
            pass

        if cls.geometry and cls.geometry not in _spell_geometry_strategies_registry:
            _spell_geometry_strategies_registry[cls.geometry] = cls

        return super().__init_subclass__()

    @classmethod
    @abstractmethod
    def build(cls, params: SpellGeometryParams) -> SpellGeometry:
        raise NotImplementedError


class LineSpell(SpellGeometryBuildStrategy):
    # TODO: Issues
    geometry = SpellGeometries.line

    @classmethod
    def build(cls, params: SpellGeometryParams) -> SpellGeometry:
        return SpellGeometry(
            x=np.arange(0, params.n),
            y=np.zeros(params.n),
        )


class PolygonSpell(SpellGeometryBuildStrategy):
    geometry = SpellGeometries.polygon

    @classmethod
    def build(cls, params: SpellGeometryParams) -> SpellGeometry:
        small_angle = np.fromiter(
            (params.t0 + i * 2 * np.pi / params.n for i in np.arange(1, params.n + 1)),
            np.float_,
        )

        return SpellGeometry(
            x=params.r * np.sin(small_angle),
            y=params.r * np.cos(small_angle),
        )


class QuadraticSpell(SpellGeometryBuildStrategy):
    geometry = SpellGeometries.quadratic

    @classmethod
    def build_from_x(cls, x: SpellGeometryAxis) -> SpellGeometry:
        return SpellGeometry(x=x, y=np.fromiter((x_**2 for x_ in x), np.float_))

    @classmethod
    def build(cls, params: SpellGeometryParams) -> SpellGeometry:
        return cls.build_from_x(
            x=np.arange(-math.floor(params.n / 2), math.ceil(params.n / 2)),
        )


class QuadraticCenteredSpell(QuadraticSpell):
    geometry = SpellGeometries.quadratic_centered

    @classmethod
    def build(cls, params: SpellGeometryParams) -> SpellGeometry:
        _x = np.array([0])
        while len(_x) < params.n:
            _x = np.append(
                _x,
                (-_x[-1] + 1) if -_x[-1] in _x else (-_x[-1]),
            )

        return cls.build_from_x(x=_x)


class CubicSpell(SpellGeometryBuildStrategy):
    geometry = SpellGeometries.cubic

    @classmethod
    def build_from_x(cls, x: SpellGeometryAxis) -> SpellGeometry:
        return SpellGeometry(
            x=x,
            y=np.fromiter((0.1 * x_**3 + -0.75 * x_ for x_ in x), np.float_),
        )

    @classmethod
    def build(cls, params: SpellGeometryParams) -> SpellGeometry:
        return cls.build_from_x(
            x=np.arange(-math.floor(params.n / 2), math.ceil(params.n / 2))
        )


class SemiCircleSpell(SpellGeometryBuildStrategy):
    geometry = SpellGeometries.semi_circle

    @classmethod
    def build(cls, params: SpellGeometryParams) -> SpellGeometry:
        theta0 = 0
        theta1 = -np.pi
        theta = np.linspace(theta0, theta1, params.n)

        return SpellGeometry(
            x=params.r * np.cos(theta),
            y=params.r * np.sin(theta),
        )


class QuarterCircleSpell(SpellGeometryBuildStrategy):
    geometry = SpellGeometries.quarter_circle

    @classmethod
    def build(cls, params: SpellGeometryParams) -> SpellGeometry:
        theta0 = 0
        theta1 = -np.pi / 2
        theta = np.linspace(theta0, theta1, params.n)

        return SpellGeometry(
            x=params.r * np.cos(theta),
            y=params.r * np.sin(theta),
        )
