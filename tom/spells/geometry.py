from __future__ import annotations

import math
import typing as t
from abc import ABC
from abc import abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from tom.util.math import Float
from tom.util.math import Int
from tom.util.math import NDArrayFloat
from tom.util.math import NDArrayInt

SpellGeometryAxis = NDArrayInt | NDArrayFloat
SpellGeometryNode = Int | Float
SpellGeometryElement = tuple[SpellGeometryNode, SpellGeometryNode]


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
        strategy: str | type[SpellGeometryBuildStrategy[P]], params: P
    ) -> SpellGeometry:
        if isinstance(strategy, str):
            # recasting for generics type checking on P
            strategy = t.cast(
                type[SpellGeometryBuildStrategy[P]],
                _spell_geometry_strategies_registry[strategy],
            )

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
    start_angle: float = 0.0
    radius: float = 1.0

    @property
    def t0(self) -> float | None:
        return self.start_angle

    @property
    def r(self) -> float:
        return self.radius


# Geometry Strategies

_spell_geometry_strategies_registry: dict[
    str, type[SpellGeometryBuildStrategy[t.Any]]
] = {}


class SpellGeometryBuildStrategy(ABC, t.Generic[P]):
    strategy: str | None = None

    def __init_subclass__(cls) -> None:
        if cls == SpellGeometryBuildStrategy:
            pass

        if cls.strategy and cls.strategy not in _spell_geometry_strategies_registry:
            _spell_geometry_strategies_registry[cls.strategy] = cls

        return super().__init_subclass__()

    @classmethod
    @abstractmethod
    def build(cls, params: P) -> SpellGeometry:
        raise NotImplementedError


class LineSpell(SpellGeometryBuildStrategy[SpellGeometryParams]):
    strategy = "line"

    @classmethod
    def build(cls, params: SpellGeometryParams) -> SpellGeometry:
        return SpellGeometry(
            x=np.arange(0, params.n),
            y=np.zeros(params.n),
        )


class PolygonSpell(SpellGeometryBuildStrategy[RadialSpellGeometryParams]):
    strategy = "polygon"

    @classmethod
    def build(cls, params: RadialSpellGeometryParams) -> SpellGeometry:
        small_angle = np.fromiter(
            (params.t0 + i * 2 * np.pi / params.n for i in np.arange(1, params.n + 1)),
            np.float_,
        )

        return SpellGeometry(
            x=params.r * np.sin(small_angle),
            y=params.r * np.cos(small_angle),
        )


class QuadraticSpell(SpellGeometryBuildStrategy[SpellGeometryParams]):
    strategy = "quadratic"

    @classmethod
    def build_from_x(cls, x: SpellGeometryAxis) -> SpellGeometry:
        return SpellGeometry(x=x, y=np.array(x_**2 for x_ in x))

    @classmethod
    def build(cls, params: SpellGeometryParams) -> SpellGeometry:
        return cls.build_from_x(
            x=np.arange(-math.floor(params.n / 2), math.ceil(params.n / 2)),
        )


class QuadraticCenteredSpell(QuadraticSpell):
    strategy = "quadratic_centered"

    @classmethod
    def build(cls, params: SpellGeometryParams) -> SpellGeometry:
        _x = np.array([0])
        while len(_x) < params.n:
            if -_x[-1] in _x:
                _x = np.append(_x, (-_x[-1] + 1) if -_x[-1] in _x else (-_x[-1]))

        return cls.build_from_x(x=_x)


class CubicSpell(SpellGeometryBuildStrategy[SpellGeometryParams]):
    strategy = "cubic"

    @classmethod
    def build_from_x(cls, x: SpellGeometryAxis) -> SpellGeometry:
        return SpellGeometry(x=x, y=np.array(0.1 * x_**3 + -0.75 * x_ for x_ in x))

    @classmethod
    def build(cls, params: SpellGeometryParams) -> SpellGeometry:
        return cls.build_from_x(
            x=np.arange(-math.floor(params.n / 2), math.ceil(params.n / 2)),
        )


class SemiCircleSpell(SpellGeometryBuildStrategy[RadialSpellGeometryParams]):
    strategy = "semi_circle"

    @classmethod
    def build(cls, params: RadialSpellGeometryParams) -> SpellGeometry:
        theta0 = 0
        theta1 = -np.pi
        theta = np.linspace(theta0, theta1, params.n)

        return SpellGeometry(
            x=params.r * np.cos(theta),
            y=params.r * np.sin(theta),
        )


class QuarterCircleSpell(SpellGeometryBuildStrategy[RadialSpellGeometryParams]):
    strategy = "quarter_circle"

    @classmethod
    def build(cls, params: RadialSpellGeometryParams) -> SpellGeometry:
        theta0 = 0
        theta1 = -np.pi / 2
        theta = np.linspace(theta0, theta1, params.n)

        return SpellGeometry(
            x=params.r * np.cos(theta),
            y=params.r * np.sin(theta),
        )
