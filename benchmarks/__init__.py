"""Planted-instance benchmark: neural networks whose local Lipschitz constant is
known by construction, used to test Lipschitz estimators at scale."""

from .planted import PlantedMeta, PlantedNet, make_planted_net

__all__ = ["PlantedMeta", "PlantedNet", "make_planted_net"]
