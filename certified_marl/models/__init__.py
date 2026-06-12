"""Actor and critic modules used by CPAC."""

__all__ = [
    "FlatEdgeActor",
    "FlatCritic",
]


def __getattr__(name: str):
    if name in {"FlatEdgeActor", "FlatCritic"}:
        from certified_marl.models import flat_actor as _fa
        return getattr(_fa, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
