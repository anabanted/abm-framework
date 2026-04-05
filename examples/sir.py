"""SIR domain: Agent factory, Rule factory, observation.

All three components are closure factories returning classes:
- sir_agent(beta, gamma) → Agent class
- sir_rule(initial_infected, perception_radius, ...) → Rule class
- grid(size, periodic) → Environment class (in grid.py)
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any, Literal, NamedTuple, Protocol, runtime_checkable

import numpy as np

from abm_framework.core import Agent, Environment, Rule

type SIRState = Literal["S", "I", "R"]


# ---------------------------------------------------------------------------
# Agent factory (closure captures beta, gamma)
# ---------------------------------------------------------------------------


def sir_agent(beta: float, gamma: float) -> type[Agent]:
    """Factory: returns an SIRAgent class with beta/gamma baked in."""

    @dataclass
    class SIRAgent(Agent):
        state: SIRState

        def next_state(
            self,
            others: Sequence[tuple[SIRAgent, float]],
            rng: np.random.Generator,
        ) -> SIRState:
            """Compute next state given other agents with distances."""
            if self.state == "S":
                for agent, _distance in others:
                    if agent.state == "I" and rng.random() < beta:
                        return "I"
                return "S"
            if self.state == "I":
                return "R" if rng.random() < gamma else "I"
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

    return SIRAgent


# ---------------------------------------------------------------------------
# Rule factory (closure captures parameters)
# ---------------------------------------------------------------------------


def sir_rule(
    initial_infected: int,
    perception_radius: float,
    move_radius: float,
    distance_to_weight: Callable[[float], float],
) -> type[Rule]:
    """Factory: returns an SIRRule class with parameters baked in."""

    _initial = initial_infected
    _perception = perception_radius
    _move = move_radius
    _d2w = distance_to_weight

    class SIRRule(Rule):

        initial_infected = _initial
        perception_radius = _perception
        move_radius = _move

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

        @staticmethod
        def init(env: Environment, agent_cls: type[Agent], rng) -> None:
            n_cells = env.size ** 2
            for _ in range(_initial):
                env.place_random(agent_cls("I"), rng)
            for _ in range(n_cells - _initial):
                env.place_random(agent_cls("S"), rng)

        @staticmethod
        def step(env: Environment, rng) -> None:
            # Phase 1: state update (synchronous)
            updates = [
                (agent, agent.next_state(
                    env.agents_with_distances(loc, _perception), rng,
                ))
                for loc, agent in env.items()
            ]
            for agent, new_state in updates:
                agent.state = new_state

            # Phase 2: movement (sequential)
            for loc, agent in list(env.items()):
                reachable = env.reachable(loc, _move)
                candidates = [(loc, 0.0)] + list(reachable)
                weights = [_d2w(d) for _, d in candidates]
                choice = agent.choose_move(weights, rng)
                chosen_loc = candidates[choice][0]
                if chosen_loc is not loc:
                    env.move(loc, chosen_loc)

    return SIRRule


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
