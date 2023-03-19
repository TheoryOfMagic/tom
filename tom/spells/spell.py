from __future__ import annotations

from typing import TypeVar

from tom.spells.features import SpellFeatures
from tom.spells.geometry import SpellGeometries
from tom.spells.geometry import SpellGeometryParams
from tom.spells.geometry import get_geometry_build_strategy
from tom.spells.pathing import SpellPathingParams
from tom.spells.pathing import SpellPathings
from tom.spells.pathing import get_pathing_build_strategy
from tom.spells.types import SpellComponentGrid
from tom.spells.types import SpellComponentNodePairs
from tom.spells.types import SpellComponentPath
from tom.spells.types import SpellGeometryStrategy
from tom.spells.types import SpellPathingStrategy


class SpellArchetype:
    geometry: SpellGeometryStrategy
    pathing: SpellPathingStrategy

    def __init__(
        self,
        geometry: SpellGeometryStrategy | SpellGeometries,
        pathing: SpellPathingStrategy | SpellPathings,
        geometry_params: SpellGeometryParams | None = None,
        pathing_params: SpellPathingParams | None = None,
    ):
        if isinstance(geometry, SpellGeometries):
            geometry = get_geometry_build_strategy(
                geometry=geometry, params=geometry_params
            )
        self.geometry = geometry

        if isinstance(pathing, SpellPathings):
            pathing = get_pathing_build_strategy(pathing=pathing, params=pathing_params)
        self.pathing = pathing

    def __call__(self, features: SpellFeatures | None = None) -> Spell:
        return self.build(features=features)

    # Factory methods

    def build(self, features: SpellFeatures | None = None) -> Spell:
        return Spell.build_with(
            geometry=self.geometry,
            pathing=self.pathing,
            features=features,
        )


S = TypeVar("S", bound="Spell")


class Spell:
    nodes: SpellComponentGrid
    paths: list[SpellComponentPath]

    def __init__(
        self,
        nodes: SpellComponentGrid,
        paths: list[SpellComponentPath] | None = None,
    ) -> None:
        self.nodes = nodes
        self.paths = paths or []

    @classmethod
    def apply_features(
        cls: type[S],
        nodes: SpellComponentGrid,
        features: SpellFeatures,
    ) -> SpellComponentNodePairs:
        # TODO: this should be handled by features
        raise NotImplementedError

    # Factory methods
    @classmethod
    def build_with(
        cls: type[S],
        geometry: SpellGeometryStrategy,
        pathing: SpellPathingStrategy,
        features: SpellFeatures | None = None,
    ) -> S:
        features = features or SpellFeatures()

        nodes = cls._build_spell_nodes_with(geometry)

        node_pairs = cls.apply_features(nodes, features)
        paths = cls._build_spell_paths_with(pathing, node_pairs)

        return cls(nodes, paths)

    @classmethod
    def _build_spell_nodes_with(
        cls,
        geometry: SpellGeometryStrategy,
    ) -> SpellComponentGrid:
        return geometry()

    @classmethod
    def _build_spell_paths_with(
        cls,
        pathing: SpellPathingStrategy,
        node_pairs: SpellComponentNodePairs | None = None,
    ) -> list[SpellComponentPath]:
        node_pairs = node_pairs or []

        return [pathing(*pair) for pair in node_pairs]
