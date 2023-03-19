from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from enum import Enum
from enum import auto

import numpy as np
from pydantic import BaseModel

from tom.spells.types import SpellComponentNode
from tom.spells.types import SpellComponentPath
from tom.spells.types import SpellPathingStrategy

# Types

SPELL_PATHING_STRATEGIES_REGISTRY: dict[SpellPathings, type[SpellPathing]] = {}


class SpellPathings(Enum):
    linear = auto()
    radial = auto()


class SpellPathingParams(BaseModel):
    class Config:
        frozen = True

    radius: float = 1.0
    offset: float = 0.0
    resolution: int = 100

    @property
    def r(self) -> float:
        return self.radius

    @property
    def b(self) -> float:
        return self.offset


# Spell Pathings


class SpellPathing(ABC):
    pathing: SpellPathings
    params: SpellPathingParams

    @property
    def p(self) -> SpellPathingParams:
        return self.params

    def __init_subclass__(cls) -> None:
        if not hasattr(cls, "pathing"):
            raise TypeError(
                "Expected non-null pathing attribute defined for SpellPathing subclasses"
            )

        if cls.pathing in SPELL_PATHING_STRATEGIES_REGISTRY:
            raise TypeError(
                f"SpellPathing subclass already defined with geometry={cls.pathing.name}"
            )

        SPELL_PATHING_STRATEGIES_REGISTRY[cls.pathing] = cls

        return super().__init_subclass__()

    def __init__(self, params: SpellPathingParams | None = None) -> None:
        self.params = params or SpellPathingParams()

    def __call__(
        self, start: SpellComponentNode, end: SpellComponentNode
    ) -> SpellComponentPath:
        return self.build(start=start, end=end)

    # Abstract factory methods

    @abstractmethod
    def build(
        self, start: SpellComponentNode, end: SpellComponentNode
    ) -> SpellComponentPath:
        raise NotImplementedError


# Spell Pathings


class LinearPath(SpellPathing):
    pathing = SpellPathings.linear

    def build(
        self, start: SpellComponentNode, end: SpellComponentNode
    ) -> SpellComponentPath:
        path: SpellComponentPath = list(
            zip(
                np.linspace(start[0], end[0], self.p.resolution, dtype=np.float_),
                np.linspace(start[1], end[1], self.p.resolution, dtype=np.float_),
                strict=True,
            )
        )

        return path


class RadialPath(SpellPathing):
    pathing = SpellPathings.radial

    def build(
        self, start: SpellComponentNode, end: SpellComponentNode
    ) -> SpellComponentPath:
        raise NotImplementedError


# Factory Methods (for use with Spell classes)


def get_pathing_build_strategy(
    pathing: SpellPathings | str,
    params: SpellPathingParams | None = None,
) -> SpellPathingStrategy:
    if isinstance(pathing, str):
        maybe_pathing = SpellPathings.__members__.get(pathing)
        if not maybe_pathing:
            raise KeyError(f"No spell pathing found for {pathing}")
        pathing = maybe_pathing

    strategy = SPELL_PATHING_STRATEGIES_REGISTRY.get(pathing)
    if not strategy:
        raise KeyError(f"No pathing strategy found for {pathing} spell pathing")

    return strategy(params)
