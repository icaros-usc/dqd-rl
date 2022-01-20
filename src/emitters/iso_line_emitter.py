"""Provides IsoLineEmitter."""
import gin
import ribs


@gin.configurable
class IsoLineEmitter(ribs.emitters.IsoLineEmitter):
    """gin-configurable version of pyribs IsoLineEmitter."""
