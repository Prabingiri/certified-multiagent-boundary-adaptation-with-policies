"""Utility modules: seeding (torch-dependent, lazy), logging, config."""

from certified_marl.utils.logging import JsonlLogger
from certified_marl.utils.registry import DotDict, get_dotted, load_config

__all__ = ["seed_everything", "JsonlLogger", "DotDict", "load_config", "get_dotted"]


def __getattr__(name: str):
    if name == "seed_everything":
        from certified_marl.utils.seeding import seed_everything as _s
        return _s
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
