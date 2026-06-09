r"""Safety counters accumulated during a rollout."""

from __future__ import annotations

from dataclasses import dataclass

from certified_marl.env.csgrag import CSGRAGState


@dataclass
class SafetyMetrics:
    """Aggregated per-rollout safety counters."""

    geom_violations: int
    ker_violations: int
    cert_violations: int
    srv_violations: int
    team_violations: int
    total_steps: int

    @property
    def total_violations(self) -> int:
        return (
            self.geom_violations
            + self.ker_violations
            + self.cert_violations
            + self.srv_violations
            + self.team_violations
        )

    def is_safe(self) -> bool:
        """True iff every counter is zero."""
        return self.total_violations == 0

    def as_dict(self) -> dict[str, int]:
        return dict(
            geom=self.geom_violations,
            ker=self.ker_violations,
            cert=self.cert_violations,
            srv=self.srv_violations,
            team=self.team_violations,
            total=self.total_violations,
            steps=self.total_steps,
        )


def safety_from_state(state: CSGRAGState) -> SafetyMetrics:
    """Extract the safety counters accumulated in `state.violations`."""
    v = state.violations
    return SafetyMetrics(
        geom_violations=v.get("geom", 0),
        ker_violations=v.get("ker", 0),
        cert_violations=v.get("cert", 0),
        srv_violations=v.get("srv", 0),
        team_violations=v.get("team", 0),
        total_steps=state.t,
    )
