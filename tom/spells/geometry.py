from __future__ import annotations

import math
from abc import ABC
from abc import abstractmethod
from enum import Enum
from enum import auto

import numpy as np
from pydantic import BaseModel

from tom.spells.types import SpellComponentAxis
from tom.spells.types import SpellComponentGrid
from tom.spells.types import SpellGeometryStrategy

# Types

SPELL_GEOMETRY_STRATEGIES_REGISTRY: dict[SpellGeometries, type[SpellGeometry]] = {}


class SpellGeometries(Enum):
    line = auto()
    polygon = auto()
    quadratic = auto()
    quadratic_centered = auto()
    cubic = auto()
    semi_circle = auto()
    quarter_circle = auto()


class SpellGeometryParams(BaseModel):
    class Config:
        frozen = True

    node_count: int = 1
    start_angle: float = 0.0
    radius: float = 1.0

    @property
    def n(self) -> int:
        return self.node_count

    @property
    def t0(self) -> float | None:
        return self.start_angle

    @property
    def r(self) -> float:
        return self.radius


# Spell Geometries


class SpellGeometry(ABC):
    geometry: SpellGeometries
    params: SpellGeometryParams

    @property
    def p(self) -> SpellGeometryParams:
        return self.params

    def __init_subclass__(cls) -> None:
        if not hasattr(cls, "geometry"):
            raise TypeError(
                "Expected non-null geometry attribute defined for SpellGeometry subclasses"
            )

        if cls.geometry in SPELL_GEOMETRY_STRATEGIES_REGISTRY:
            raise TypeError(
                f"SpellGeometryBuildStrategy subclass already defined with geometry={cls.geometry.name}"
            )

        SPELL_GEOMETRY_STRATEGIES_REGISTRY[cls.geometry] = cls

    def __init__(self, params: SpellGeometryParams | None = None):
        self.params = params or SpellGeometryParams()

    def __call__(self) -> SpellComponentGrid:
        return self.build()

    # Abstract factory methods

    @abstractmethod
    def build(self) -> SpellComponentGrid:
        raise NotImplementedError


class LineSpellGeometry(SpellGeometry):
    geometry = SpellGeometries.line

    def build(self) -> SpellComponentGrid:
        return (
            np.arange(0, self.p.n),
            np.zeros(self.p.n),
        )


class PolygonSpellGeometry(SpellGeometry):
    geometry = SpellGeometries.polygon

    def build(self) -> SpellComponentGrid:
        small_angle = np.fromiter(
            (self.p.t0 + i * 2 * np.pi / self.p.n for i in np.arange(1, self.p.n + 1)),
            np.float_,
        )

        return (
            self.p.r * np.sin(small_angle),
            self.p.r * np.cos(small_angle),
        )


class QuadraticSpellGeometry(SpellGeometry):
    geometry = SpellGeometries.quadratic

    def build(self) -> SpellComponentGrid:
        return self._build_from_x(
            x=np.arange(-math.floor(self.p.n / 2), math.ceil(self.p.n / 2)),
        )

    def _build_from_x(self, x: SpellComponentAxis) -> SpellComponentGrid:
        return (
            x,
            np.fromiter((x_**2 for x_ in x), np.float_),
        )


class QuadraticCenteredSpell(QuadraticSpellGeometry):
    geometry = SpellGeometries.quadratic_centered

    def build(self) -> SpellComponentGrid:
        _x = np.array([0])
        while len(_x) < self.p.n:
            _x = np.append(
                _x,
                (-_x[-1] + 1) if -_x[-1] in _x else (-_x[-1]),
            )

        return self._build_from_x(x=_x)


class CubicSpellGeometry(SpellGeometry):
    geometry = SpellGeometries.cubic

    def build(self) -> SpellComponentGrid:
        return self._build_from_x(
            x=np.arange(-math.floor(self.p.n / 2), math.ceil(self.p.n / 2))
        )

    def _build_from_x(self, x: SpellComponentAxis) -> SpellComponentGrid:
        return (
            x,
            np.fromiter((0.1 * x_**3 + -0.75 * x_ for x_ in x), np.float_),
        )


class SemiCircleSpellGeometry(SpellGeometry):
    geometry = SpellGeometries.semi_circle

    def build(self) -> SpellComponentGrid:
        theta0 = 0
        theta1 = -np.pi
        theta = np.linspace(theta0, theta1, self.p.n)

        return (
            self.p.r * np.cos(theta),
            self.p.r * np.sin(theta),
        )


class QuarterCircleSpellGeometry(SpellGeometry):
    geometry = SpellGeometries.quarter_circle

    def build(self) -> SpellComponentGrid:
        theta0 = 0
        theta1 = -np.pi / 2
        theta = np.linspace(theta0, theta1, self.p.n)

        return (
            self.p.r * np.cos(theta),
            self.p.r * np.sin(theta),
        )


# Factory Methods (for use with Spell classes)


def get_geometry_build_strategy(
    geometry: SpellGeometries | str,
    params: SpellGeometryParams | None = None,
) -> SpellGeometryStrategy:
    if isinstance(geometry, str):
        maybe_geometry = SpellGeometries.__members__.get(geometry)
        if not maybe_geometry:
            raise KeyError(f"No spell geometry found for {geometry}")
        geometry = maybe_geometry

    strategy = SPELL_GEOMETRY_STRATEGIES_REGISTRY.get(geometry)
    if not strategy:
        raise KeyError(f"No build strategy found for {geometry} spell geometry")

    return strategy(params)
