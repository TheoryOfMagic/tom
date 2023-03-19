from __future__ import annotations

from collections.abc import Callable

from tom.util.math import Float
from tom.util.math import Int
from tom.util.math import NDArrayFloat
from tom.util.math import NDArrayInt

SpellComponentElement = Int | Float
SpellComponentNode = tuple[SpellComponentElement, SpellComponentElement]
SpellComponentNodePairs = list[tuple[SpellComponentNode, SpellComponentNode]]
SpellComponentPath = list[SpellComponentNode]
SpellComponentAxis = NDArrayInt | NDArrayFloat
SpellComponentGrid = tuple[SpellComponentAxis, SpellComponentAxis]


SpellGeometryStrategy = Callable[[], SpellComponentGrid]
SpellPathingStrategy = Callable[
    [SpellComponentNode, SpellComponentNode], SpellComponentPath
]
