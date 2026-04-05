"""SIR domain: Agent, Rule, observation."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, NamedTuple, Protocol, runtime_checkable

import numpy as np

from abm_framework.core import Agent, Environment, Rule

type SIRState = Literal["S", "I", "R"]


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


@dataclass
class SIRAgent(Agent):
    """SIR agent. Receives (agent, intensity) contacts, decides infection."""

    state: SIRState
    beta: float
    gamma: float

    def next_state(
        self,
        contacts: Sequence[tuple[SIRAgent, float]],
        rng: np.random.Generator,
    ) -> SIRState:
        """Compute next state from contacts (agent, intensity) pairs."""
        if self.state == "S":
            for agent, intensity in contacts:
                if agent.state == "I" and rng.random() < self.beta * intensity:
                    return "I"
            return "S"
        if self.state == "I":
            return "R" if rng.random() < self.gamma else "I"
        return self.state

    def move(self, env: Environment, loc: Any, rng: np.random.Generator) -> None:
        """Delegate movement to environment's move rule."""
        env.try_move(loc, self, rng)


# ---------------------------------------------------------------------------
# Rule
# ---------------------------------------------------------------------------


class SIRRule(Rule):
    """SIR transition rule. Thin orchestrator.

    Phase 1: infection — env.contacts → agent.next_state (synchronous)
    Phase 2: movement — agent.move → env.try_move (sequential)
    """

    @runtime_checkable
    class EnvSpec(Protocol):
        def items(self) -> Any: ...
        def contacts(self, location: Any) -> Sequence[tuple[Any, float]]: ...
        def try_move(self, location: Any, agent: Any, rng: Any) -> None: ...
        def place_random(self, agent: Any, rng: Any) -> None: ...

    @runtime_checkable
    class AgentSpec(Protocol):
        state: SIRState
        def next_state(
            self, contacts: Sequence[tuple[Any, float]],
            rng: np.random.Generator,
        ) -> SIRState: ...
        def move(self, env: Any, loc: Any, rng: np.random.Generator) -> None: ...

    def init(self, env: Environment, agents: Sequence[Agent], rng) -> None:
        for agent in agents:
            env.place_random(agent, rng)

    def step(self, env: Environment, rng) -> None:
        # Phase 1: infection + recovery (synchronous)
        updates = [
            (agent, agent.next_state(env.contacts(loc), rng))
            for loc, agent in env.items()
        ]
        for agent, new_state in updates:
            agent.state = new_state

        # Phase 2: movement (sequential) — Rule → Agent → Env
        for loc, agent in list(env.items()):
            agent.move(env, loc, rng)


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
