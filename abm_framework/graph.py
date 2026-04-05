"""Graph: pure spatial structure. Nodes + edges, multiple agents per node."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from .core import Agent, Environment


class Graph(Environment):
    """Graph environment. Multiple agents can share a node."""

    def __init__(self, n_nodes: int, edges: list[tuple[int, int]]) -> None:
        self.n_nodes = n_nodes
        self.adjacency: dict[int, list[int]] = {i: [] for i in range(n_nodes)}
        for a, b in edges:
            self.adjacency[a].append(b)
            self.adjacency[b].append(a)
        self._node_agents: dict[int, list[Agent]] = {i: [] for i in range(n_nodes)}

    def place_random(self, agent: Agent, rng: np.random.Generator) -> None:
        node = int(rng.integers(self.n_nodes))
        self._node_agents[node].append(agent)

    def __iter__(self) -> Iterator:
        for agents in self._node_agents.values():
            yield from agents

    def items(self) -> Iterator[tuple[int, Agent]]:
        for node, agents in self._node_agents.items():
            for agent in agents:
                yield (node, agent)

    def agents_at(self, node: int) -> list[Agent]:
        return self._node_agents[node]

    def move_agent(self, from_node: int, agent: Agent, to_node: int) -> None:
        self._node_agents[from_node].remove(agent)
        self._node_agents[to_node].append(agent)
