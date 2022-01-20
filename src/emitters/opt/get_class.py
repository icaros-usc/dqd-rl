"""Provides method for retrieving opt class from string name."""
from src.emitters.opt.cma_es import CMAEvolutionStrategy

CLASSES = {
    "cma_es": CMAEvolutionStrategy,
}


def get_class(name):
    """Retrieves opt class associated with a name."""
    if name in CLASSES:
        return CLASSES[name]
    raise ValueError(f"Unknown es '{name}'")
