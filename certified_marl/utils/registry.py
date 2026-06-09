"""Config loading: YAML files -> nested dict with dotted-key access."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class DotDict(dict):
    """Dict with attribute access and dotted-key lookup."""

    def __getattr__(self, k: str) -> Any:
        try:
            v = self[k]
        except KeyError as e:
            raise AttributeError(str(e))
        return DotDict(v) if isinstance(v, dict) else v

    def __setattr__(self, k: str, v: Any) -> None:
        self[k] = v


def load_config(path: str | Path) -> DotDict:
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return DotDict(data or {})


def get_dotted(cfg: DotDict, key: str, default: Any = None) -> Any:
    """`get_dotted(cfg, 'env.arrival.rate')` -> nested lookup."""
    parts = key.split(".")
    cur: Any = cfg
    for p in parts:
        if isinstance(cur, dict) and p in cur:
            cur = cur[p]
        else:
            return default
    return cur
