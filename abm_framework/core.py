"""ABM Framework core: Environment, Agent, Rule markers + Model assembler."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence


class Environment(ABC):
    """Marker base for all spatial structures."""


class Agent(ABC):
    """Marker base for all agents."""


class Rule(ABC):
    """Transition rule.

    Subclasses define:
    - EnvSpec: Protocol for environment requirements
    - AgentSpec: Protocol for agent requirements
    - init(env, agents, rng): place agents into the environment
    - step(env, rng): advance one timestep
    """

    @abstractmethod
    def init(self, env: Environment, agents: Sequence[Agent], rng) -> None:
        """Place agents into the environment."""

    @abstractmethod
    def step(self, env: Environment, rng) -> None:
        """Advance simulation by one timestep."""


class Model:
    """Assembles Rule + Environment + Agents.

    All three are plain instances. Model verifies compatibility
    and provides step/run.
    """

    def __init__(
        self,
        rule: Rule,
        env: Environment,
        agents: Sequence[Agent],
        *,
        rng: object,
    ) -> None:
        self.rule = rule
        self.env = env
        self.rng = rng

        self._check_env(rule, env)
        self._check_agents(rule, agents)
        rule.init(env, agents, rng)

    @staticmethod
    def _check_env(rule: Rule, env: Environment) -> None:
        env_spec = getattr(type(rule), "EnvSpec", None)
        if env_spec and not isinstance(env, env_spec):
            raise TypeError(
                f"{type(env).__name__} does not satisfy "
                f"{type(rule).__name__}.EnvSpec"
            )

    @staticmethod
    def _check_agents(rule: Rule, agents: Sequence[Agent]) -> None:
        agent_spec = getattr(type(rule), "AgentSpec", None)
        if not agent_spec:
            return
        for agent in agents:
            if not isinstance(agent, Agent):
                raise TypeError(f"{agent!r} is not an Agent subclass")
            if not isinstance(agent, agent_spec):
                raise TypeError(
                    f"{agent!r} does not satisfy "
                    f"{type(rule).__name__}.AgentSpec"
                )
            break

    def step(self) -> None:
        self.rule.step(self.env, self.rng)

    def run[O](
        self, n_steps: int, observe: Callable[[Environment], O],
    ) -> list[O]:
        observations = [observe(self.env)]
        for _ in range(n_steps):
            self.step()
            observations.append(observe(self.env))
        return observations
