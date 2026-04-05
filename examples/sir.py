"""SIR domain: Agent factory, Rule, observation."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Iterator
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
            nearby: Sequence[tuple[SIRAgent, float]],
            rng: np.random.Generator,
        ) -> SIRState:
            """Compute next state from current state and nearby agents."""
            if self.state == "S":
                for agent, _distance in nearby:
                    if agent.state == "I" and rng.random() < beta:
                        return "I"
                return "S"
            if self.state == "I":
                return "R" if rng.random() < gamma else "I"
            return self.state

    return SIRAgent


# ---------------------------------------------------------------------------
# Rule (class, not instantiated)
# ---------------------------------------------------------------------------


class SIRRule(Rule):
    """SIR transition rule.

    Defines:
    - EnvSpec: environment must have items(), nearby(), place_random()
    - AgentSpec: agents must have state and next_state(nearby, rng)
    - init(): populate grid with initial infected + susceptible
    - step(): synchronous comonadic update
    """

    INITIAL_INFECTED: int = 3
    PERCEPTION_RADIUS: float = 1.5

    @runtime_checkable
    class EnvSpec(Protocol):
        """Environment capabilities required by this rule."""

        def items(self) -> Iterator[tuple[Any, Any]]: ...
        def nearby(self, location: Any, radius: float) -> Sequence[tuple[Any, float]]: ...
        def place_random(self, agent: Any, rng: Any) -> None: ...

    @runtime_checkable
    class AgentSpec(Protocol):
        """Agent capabilities required by this rule."""

        state: SIRState

        def next_state(
            self,
            nearby: Sequence[tuple[Any, float]],
            rng: np.random.Generator,
        ) -> SIRState: ...

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
        """Synchronous update: compute all next states, then apply."""
        updates = [
            (agent, agent.next_state(
                env.nearby(loc, SIRRule.PERCEPTION_RADIUS), rng,
            ))
            for loc, agent in env.items()
        ]
        for agent, new_state in updates:
            agent.state = new_state


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
