"""Provides GaussianEmitter."""
import gin
import ribs


@gin.configurable
class GaussianEmitter(ribs.emitters.GaussianEmitter):
    """gin-configurable version of pyribs GaussianEmitter."""
