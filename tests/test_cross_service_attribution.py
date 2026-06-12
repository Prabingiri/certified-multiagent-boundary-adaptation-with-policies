from __future__ import annotations

import numpy as np

from certified_marl.env.arrivals import ArrivalProcess, Event
from certified_marl.env.csgrag import CSGRAGEnv
from certified_marl.env.geometry import Rect
from certified_marl.metrics.panel import ExperimentPanel


class OneShotArrival(ArrivalProcess):
    def __init__(self, event: Event):
        super().__init__(np.random.default_rng(0))
        self.event = event
        self.used = False

    def step(self, t: float, dt: float) -> list[Event]:
        if self.used:
            return []
        self.used = True
        return [self.event]


def _two_region_env(event: Event) -> CSGRAGEnv:
    return CSGRAGEnv(
        regions=[
            Rect(0.0, 0.0, 4.0, 4.0),
            Rect(4.0, 0.0, 8.0, 4.0),
        ],
        kernels=[
            Rect(0.5, 0.5, 2.9, 3.5),
            Rect(5.1, 0.5, 7.5, 3.5),
        ],
        U_bar=[10.0, 10.0],
        arrivals=OneShotArrival(event),
        obstacles=[],
        bounds=(0.0, 0.0, 8.0, 4.0),
        dt=1.0,
        speed=1.0,
        delta_star_default=1.0,
        delta_step=0.1,
        rng=np.random.default_rng(0),
        horizon=1,
        initial_delta_fraction=1.0,
    )


def test_true_owner_cross_admission_is_separate_from_buffer_admission():
    env = _two_region_env(Event(x=3.9, y=2.0, occurrence_time=0.0))
    state = env.reset()
    owner = state.agents[0]
    owner.queue = [
        Event(x=1.0, y=1.0, occurrence_time=0.0),
        Event(x=1.0, y=3.0, occurrence_time=0.0),
    ]

    state, info = env.step({(0, 1): 0})

    assert info["buffer_admitted_by_edge"][(0, 1)] == 1
    assert info["cross_admitted_by_edge"][(0, 1)] == 1
    assert info["owner_cross_admitted_by_edge"][(0, 1)] == 1
    assert info["same_owner_buffer_admitted_by_edge"].get((0, 1), 0) == 0


def test_same_owner_buffer_admission_does_not_count_as_cross_service():
    env = _two_region_env(Event(x=3.9, y=2.0, occurrence_time=0.0))
    state = env.reset()
    neighbor = state.agents[1]
    neighbor.queue = [
        Event(x=7.0, y=1.0, occurrence_time=0.0),
        Event(x=7.0, y=3.0, occurrence_time=0.0),
    ]

    state, info = env.step({(0, 1): 0})

    assert info["buffer_admitted_by_edge"][(0, 1)] == 1
    assert info["same_owner_buffer_admitted_by_edge"][(0, 1)] == 1
    assert info["cross_admitted_by_edge"].get((0, 1), 0) == 0
    assert info["owner_cross_admitted_by_edge"].get((0, 1), 0) == 0


def test_panel_records_true_cross_and_buffer_path_counts():
    env = _two_region_env(Event(x=3.9, y=2.0, occurrence_time=0.0))
    state = env.reset()
    state.agents[0].queue = [
        Event(x=1.0, y=1.0, occurrence_time=0.0),
        Event(x=1.0, y=3.0, occurrence_time=0.0),
    ]
    panel = ExperimentPanel()
    panel.start()

    state, info = env.step({(0, 1): 0})
    panel.record(state, info)
    results = panel.finalize(state)
    throughput = results.throughput.as_dict()

    assert throughput["cross_admitted_total"] == 1
    assert throughput["buffer_admitted_total"] == 1
    assert throughput["same_owner_buffer_admitted_total"] == 0
    assert throughput["buffer_rejected_total"] == 0
