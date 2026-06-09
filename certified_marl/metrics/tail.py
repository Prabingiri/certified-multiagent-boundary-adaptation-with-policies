r"""Response-time summaries for completed events."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass
class TailMetrics:
    """Reported response-time statistics."""

    n: int
    mean: float
    p95: float
    max: float

    def as_dict(self) -> dict[str, float]:
        return dict(n=self.n, mean=self.mean, p95=self.p95, max=self.max)


def tail_metrics(times: Iterable[float], alpha: float = 0.95) -> TailMetrics:
    """Compute mean, p95, and max response time."""
    x = np.asarray(list(times), dtype=float)
    if x.size == 0:
        return TailMetrics(0, 0.0, 0.0, 0.0)
    return TailMetrics(
        n=int(x.size),
        mean=float(x.mean()),
        p95=float(np.quantile(x, alpha)),
        max=float(x.max()),
    )
