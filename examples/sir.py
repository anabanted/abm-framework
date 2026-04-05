"""SIR domain: Agent, Rule, observation. All plain classes and instances."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any, Literal, NamedTuple, Protocol, runtime_checkable

import numpy as np

from abm_framework.core import Agent, Environment, Rule

type SIRState = Literal["S", "I", "R"]


# ---------------------------------------------------------------------------
# Agent (plain dataclass, parameters per instance)
# ---------------------------------------------------------------------------


@dataclass
class SIRAgent(Agent):
    """SIR agent with per-instance parameters."""

    state: SIRState
    beta: float
    gamma: float

    def next_state(
        self,
        others: Sequence[tuple[SIRAgent, float]],
        rng: np.random.Generator,
    ) -> SIRState:
        """Compute next state given other agents with distances."""
        if self.state == "S":
            for agent, _distance in others:
                if agent.state == "I" and rng.random() < self.beta:
                    return "I"
            return "S"
        if self.state == "I":
            return "R" if rng.random() < self.gamma else "I"
        return self.state

    def choose_move(
        self,
        weights: Sequence[float],
        rng: np.random.Generator,
    ) -> int:
        """Choose destination by index. weights[0] = current location."""
        if len(weights) == 1:
            return 0
        total = sum(weights)
        if total == 0:
            return 0
        probs = [w / total for w in weights]
        return int(rng.choice(len(weights), p=probs))


# ---------------------------------------------------------------------------
# Rule (plain instance with parameters)
# ---------------------------------------------------------------------------


class SIRRule(Rule):
    """SIR transition rule."""

    @runtime_checkable
    class EnvSpec(Protocol):
        def items(self) -> Iterator[tuple[Any, Any]]: ...
        def agents_with_distances(
            self, location: Any, max_distance: float,
        ) -> Sequence[tuple[Any, float]]: ...
        def reachable(
            self, location: Any, max_distance: float,
        ) -> Sequence[tuple[Any, float]]: ...
        def move(self, from_loc: Any, to_loc: Any) -> None: ...
        def place_random(self, agent: Any, rng: Any) -> None: ...

    @runtime_checkable
    class AgentSpec(Protocol):
        state: SIRState
        def next_state(
            self, others: Sequence[tuple[Any, float]],
            rng: np.random.Generator,
        ) -> SIRState: ...
        def choose_move(
            self, weights: Sequence[float],
            rng: np.random.Generator,
        ) -> int: ...

    def __init__(
        self,
        perception_radius: float,
        move_radius: float,
        distance_to_weight: Callable[[float], float],
    ) -> None:
        self.perception_radius = perception_radius
        self.move_radius = move_radius
        self.distance_to_weight = distance_to_weight

    def init(self, env: Environment, agents: Sequence[Agent], rng) -> None:
        """Place agents into the environment."""
        for agent in agents:
            env.place_random(agent, rng)

    def step(self, env: Environment, rng) -> None:
        """Synchronous state update + sequential movement."""
        # Phase 1: state update (synchronous)
        updates = [
            (agent, agent.next_state(
                env.agents_with_distances(loc, self.perception_radius), rng,
            ))
            for loc, agent in env.items()
        ]
        for agent, new_state in updates:
            agent.state = new_state

        # Phase 2: movement (sequential)
        for loc, agent in list(env.items()):
            reachable = env.reachable(loc, self.move_radius)
            candidates = [(loc, 0.0)] + list(reachable)
            weights = [self.distance_to_weight(d) for _, d in candidates]
            choice = agent.choose_move(weights, rng)
            chosen_loc = candidates[choice][0]
            if chosen_loc is not loc:
                env.move(loc, chosen_loc)


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------


class SIRCounts(NamedTuple):
    s: int
    i: int
    r: int


def count_by_state(env: Environment) -> SIRCounts:
    counts = Counter(a.state for a in env)
    return SIRCounts(s=counts["S"], i=counts["I"], r=counts["R"])
