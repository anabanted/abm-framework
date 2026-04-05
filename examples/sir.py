"""SIR domain: Agent factory, Rule, observation."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Literal, NamedTuple, Protocol, Sequence, runtime_checkable

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
        """SIR agent. beta/gamma captured from closure."""

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
            current_loc: Any,
            destinations: Sequence[tuple[Any, float]],
            rng: np.random.Generator,
        ) -> object:
            """Choose where to move. Always returns a location (may be current)."""
            if not destinations:
                return current_loc  # no alternatives → stay in place
            loc, _dist = destinations[rng.integers(len(destinations))]
            return loc

    return SIRAgent


# ---------------------------------------------------------------------------
# Rule (class, not instantiated)
# ---------------------------------------------------------------------------


class SIRRule(Rule):
    """SIR transition rule.

    Requires from Environment:
    - items(), agents_with_distances(loc, max_dist), reachable(loc, max_dist),
      move(from, to), place_random(agent, rng)

    Requires from Agent:
    - state, next_state(others, rng), choose_move(destinations, rng)
    """

    INITIAL_INFECTED: int = 3
    PERCEPTION_RADIUS: float = 1.5
    MOVE_RADIUS: float = 1.5

    @runtime_checkable
    class EnvSpec(Protocol):
        """Environment capabilities required by this rule."""

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
        """Agent capabilities required by this rule."""

        state: SIRState

        def next_state(
            self,
            others: Sequence[tuple[Any, float]],
            rng: np.random.Generator,
        ) -> SIRState: ...

        def choose_move(
            self,
            current_loc: Any,
            destinations: Sequence[tuple[Any, float]],
            rng: np.random.Generator,
        ) -> object: ...

    @staticmethod
    def init(env: Environment, agent_cls: type[Agent], rng) -> None:
        """Populate environment with initial SIR agents."""
        n_cells = env.size ** 2
        for _ in range(SIRRule.INITIAL_INFECTED):
            env.place_random(agent_cls("I"), rng)
        for _ in range(n_cells - SIRRule.INITIAL_INFECTED):
            env.place_random(agent_cls("S"), rng)

    @staticmethod
    def step(env: Environment, rng) -> None:
        """Synchronous state update + sequential movement."""
        # Phase 1: compute all next states (synchronous)
        updates = [
            (agent, agent.next_state(
                env.agents_with_distances(loc, SIRRule.PERCEPTION_RADIUS), rng,
            ))
            for loc, agent in env.items()
        ]
        for agent, new_state in updates:
            agent.state = new_state

        # Phase 2: movement (sequential — order matters for occupancy)
        for loc, agent in list(env.items()):
            destinations = env.reachable(loc, SIRRule.MOVE_RADIUS)
            dest = agent.choose_move(loc, destinations, rng)
            if dest is not loc:
                env.move(loc, dest)


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------


class SIRCounts(NamedTuple):
    """Counts of S, I, R agents at a single timestep."""

    s: int
    i: int
    r: int


def count_by_state(env: Environment) -> SIRCounts:
    """Count agents in each SIR state."""
    counts = Counter(a.state for a in env)
    return SIRCounts(s=counts["S"], i=counts["I"], r=counts["R"])
