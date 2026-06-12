r"""Load-signal dispersion metric."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from certified_marl.env.csgrag import CSGRAGState


@dataclass
class ImbalanceMetrics:
    """Reported load-dispersion summary over a rollout."""

    mean_coef_var: float

    def as_dict(self) -> dict[str, float]:
        return dict(mean_coef_var=self.mean_coef_var)


def coef_variation(loads: Iterable[float]) -> float:
    """Coefficient of variation, with zero returned for zero mean."""
    x = np.asarray(list(loads), dtype=float)
    if x.size == 0:
        return 0.0
    mu = float(x.mean())
    if abs(mu) < 1e-15:
        return 0.0
    return float(x.std() / mu)


class ImbalanceTracker:
    """Records load coefficient of variation over a rollout."""

    def __init__(self):
        self._cv_trace: list[float] = []

    def record(self, state: CSGRAGState) -> None:
        loads = [a.load_pressure_ewma for a in state.agents]
        self._cv_trace.append(coef_variation(loads))

    def finalize(self) -> ImbalanceMetrics:
        c = np.asarray(self._cv_trace) if self._cv_trace else np.zeros(1)
        return ImbalanceMetrics(mean_coef_var=float(c.mean()))

    @property
    def coef_var_trace(self) -> list[float]:
        return list(self._cv_trace)


__all__ = [
    "ImbalanceMetrics",
    "ImbalanceTracker",
    "coef_variation",
]
