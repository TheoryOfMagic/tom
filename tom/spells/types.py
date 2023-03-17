from __future__ import annotations

from enum import Enum
from enum import auto

from pydantic import BaseModel

SpellRanges: list[int] = [0, 5, 10, 30, 60, 90, 100, 150, 300, 500, 1000, 5280]
SpellLevels: list[int] = list(range(0, 12))


class IndexedEnum(Enum):
    @classmethod
    def all(cls) -> list[str]:
        return list(cls._member_names_)

    def index(self) -> int:
        return sorted(self.all()).index(self.name)


class SpellRange(IndexedEnum):
    r_0 = 0
    r_5 = 5
    r_10 = 10
    r_30 = 30
    r_60 = 60
    r_90 = 90
    r_100 = 100
    r_150 = 150
    r_300 = 300
    r_500 = 500
    r_1000 = 1000
    r_5280 = 5280


class SpellAreaType(IndexedEnum):
    single_target = auto()  # TODO: What is this?
    multi_target = auto()  # TODO: What is this?
    wall = auto()
    sphere = auto()
    cylinder = auto()
    cone = auto()
    line = auto()
    cube = auto()
    square = auto()
    circle = auto()


class SpellDamageType(IndexedEnum):
    bludgeoning = auto()
    cold = auto()
    poison = auto()
    thunder = auto()
    acid = auto()
    necrotic = auto()
    force = auto()
    radiant = auto()
    psychic = auto()
    lightning = auto()
    fire = auto()
    slashing = auto()


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
