r"""Metric aggregation for one evaluation rollout."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Optional

from certified_marl.env.csgrag import CSGRAGState
from certified_marl.metrics.imbalance import ImbalanceMetrics, ImbalanceTracker
from certified_marl.metrics.safety import SafetyMetrics, safety_from_state
from certified_marl.metrics.tail import TailMetrics, tail_metrics


@dataclass
class ThroughputMetrics:
    """Admission and cross-service statistics."""

    admitted: int
    rejected: int
    completed: int
    cross_admitted_total: int = 0
    buffer_admitted_total: int = 0
    same_owner_buffer_admitted_total: int = 0
    buffer_rejected_total: int = 0

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class ExperimentResults:
    """Reported metrics from one evaluation rollout."""

    safety: SafetyMetrics
    tail: Optional[TailMetrics]
    imbalance: ImbalanceMetrics
    throughput: ThroughputMetrics

    def as_dict(self) -> dict:
        return dict(
            safety=self.safety.as_dict(),
            tail=self.tail.as_dict() if self.tail else None,
            imbalance=self.imbalance.as_dict(),
            throughput=self.throughput.as_dict(),
        )

    def to_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.as_dict(), f, indent=2)

    def print_summary(self) -> None:
        d = self.as_dict()
        print("=" * 70)
        print("EXPERIMENT RESULTS")
        print("=" * 70)
        for section in ("safety", "tail", "imbalance", "throughput"):
            if d[section] is None:
                continue
            print(f"\n{section.upper()}")
            for k, v in d[section].items():
                print(f"    {k:24s} = {v}")
        print("=" * 70)


class ExperimentPanel:
    """Online metric recorder for one rollout."""

    def __init__(self):
        self._imb = ImbalanceTracker()
        self._admitted = 0
        self._rejected = 0
        self._completed = 0
        self._cross_admitted = 0
        self._buffer_admitted = 0
        self._same_owner_buffer_admitted = 0
        self._buffer_rejected = 0

    def start(self) -> None:
        pass

    def record(self, state: CSGRAGState, info: dict) -> None:
        self._imb.record(state)
        self._admitted += info.get("admitted", 0)
        self._rejected += info.get("rejected", 0)
        self._completed += info.get("completed", 0)
        cross_by_edge = info.get("owner_cross_admitted_by_edge", info.get("cross_admitted_by_edge", {}))
        if isinstance(cross_by_edge, dict):
            self._cross_admitted += sum(int(v) for v in cross_by_edge.values())
        buffer_by_edge = info.get("buffer_admitted_by_edge", {})
        if isinstance(buffer_by_edge, dict):
            self._buffer_admitted += sum(int(v) for v in buffer_by_edge.values())
        same_owner_by_edge = info.get("same_owner_buffer_admitted_by_edge", {})
        if isinstance(same_owner_by_edge, dict):
            self._same_owner_buffer_admitted += sum(int(v) for v in same_owner_by_edge.values())
        buffer_rejected_by_edge = info.get("buffer_rejected_by_edge", {})
        if isinstance(buffer_rejected_by_edge, dict):
            self._buffer_rejected += sum(int(v) for v in buffer_rejected_by_edge.values())

    def finalize(self, state: CSGRAGState) -> ExperimentResults:
        throughput = ThroughputMetrics(
            admitted=self._admitted,
            rejected=self._rejected,
            completed=self._completed,
            cross_admitted_total=self._cross_admitted,
            buffer_admitted_total=self._buffer_admitted,
            same_owner_buffer_admitted_total=self._same_owner_buffer_admitted,
            buffer_rejected_total=self._buffer_rejected,
        )
        tail = tail_metrics(state.response_times, alpha=0.95) if state.response_times else None
        return ExperimentResults(
            safety=safety_from_state(state),
            tail=tail,
            imbalance=self._imb.finalize(),
            throughput=throughput,
        )


__all__ = ["ExperimentPanel", "ExperimentResults", "ThroughputMetrics"]
