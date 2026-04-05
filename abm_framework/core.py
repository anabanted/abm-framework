"""ABM Framework core: Environment, Agent, Rule markers + Model assembler."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable


class Environment(ABC):
    """Marker base for all spatial structures."""


class Agent(ABC):
    """Marker base for all agents."""


class Rule(ABC):
    """Transition rule (used as a class, not instantiated directly).

    Subclasses define:
    - EnvSpec: Protocol specifying environment requirements
    - AgentSpec: Protocol specifying agent requirements
    - init(env, agent_cls, rng): populate environment with agents
    - step(env, rng): advance one timestep
    """

    @staticmethod
    @abstractmethod
    def init(env: Environment, agent_cls: type[Agent], rng) -> None:
        """Populate the environment with agents."""

    @staticmethod
    @abstractmethod
    def step(env: Environment, rng) -> None:
        """Advance simulation by one timestep."""


class Model:
    """Assembles Rule + Agent + Environment.

    Receives three classes (not instances for Agent/Rule),
    verifies compatibility, and creates a runnable simulation.
    """

    def __init__(
        self,
        rule: type[Rule],
        agent_cls: type[Agent],
        env_cls: type[Environment],
        *,
        rng: object,
    ) -> None:
        self.rule = rule
        self.agent_cls = agent_cls
        self.rng = rng

        # Verify environment compatibility
        self._check_env(rule, env_cls)

        # Instantiate environment and populate with agents
        self.env: Environment = env_cls()
        rule.init(self.env, agent_cls, rng)

        # Verify agent compatibility (after init populated the env)
        self._check_agents(rule, self.env)

    @staticmethod
    def _check_env(rule: type[Rule], env_cls: type[Environment]) -> None:
        env_spec = getattr(rule, "EnvSpec", None)
        if env_spec is None:
            return
        # Create a temporary instance to check Protocol conformance
        tmp = env_cls()
        if not isinstance(tmp, env_spec):
            raise TypeError(
                f"{env_cls.__name__} does not satisfy "
                f"{rule.__name__}.EnvSpec"
            )

    @staticmethod
    def _check_agents(rule: type[Rule], env: Environment) -> None:
        agent_spec = getattr(rule, "AgentSpec", None)
        if agent_spec is None:
            return
        for agent in env:
            if not isinstance(agent, Agent):
                raise TypeError(f"{agent!r} is not an Agent subclass")
            if not isinstance(agent, agent_spec):
                raise TypeError(
                    f"{agent!r} does not satisfy "
                    f"{rule.__name__}.AgentSpec"
                )
            break  # check first agent only

    def step(self) -> None:
        """Advance simulation by one timestep."""
        self.rule.step(self.env, self.rng)

    def run[O](
        self, n_steps: int, observe: Callable[[Environment], O],
    ) -> list[O]:
        """Step and observe repeatedly."""
        observations = [observe(self.env)]
        for _ in range(n_steps):
            self.step()
            observations.append(observe(self.env))
        return observations
