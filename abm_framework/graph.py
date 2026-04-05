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

    def edges(self) -> list[tuple[int, int]]:
        """Unique undirected edges."""
        return [(i, j) for i, nbrs in self.adjacency.items()
                for j in nbrs if i < j]

    def spring_layout(
        self, rng: np.random.Generator, n_iter: int = 100,
        k: float | None = None,
    ) -> dict[int, tuple[float, float]]:
        """Fruchterman-Reingold spring layout."""
        n = self.n_nodes
        if k is None:
            k = 1.0 / np.sqrt(n)

        pos = rng.uniform(-0.5, 0.5, (n, 2))
        edges = self.edges()

        for iteration in range(n_iter):
            temp = 0.1 * (1.0 - iteration / n_iter)
            disp = np.zeros((n, 2))

            # Repulsion between all pairs
            for i in range(n):
                diff = pos[i] - pos
                dist = np.sqrt((diff ** 2).sum(axis=1))
                dist[i] = 1.0  # avoid division by zero
                force = (k * k / dist)[:, np.newaxis] * diff / dist[:, np.newaxis]
                disp[i] += force.sum(axis=0)

            # Attraction along edges
            for i, j in edges:
                diff = pos[j] - pos[i]
                dist = max(np.linalg.norm(diff), 1e-6)
                f = (dist / k) * diff / dist
                disp[i] += f
                disp[j] -= f

            # Apply with temperature
            mag = np.sqrt((disp ** 2).sum(axis=1))
            mag = np.maximum(mag, 1e-6)
            pos += disp / mag[:, np.newaxis] * np.minimum(temp, mag)[:, np.newaxis]

        # Normalize to [-1, 1]
        pos -= pos.mean(axis=0)
        scale = np.abs(pos).max()
        if scale > 0:
            pos /= scale

        return {i: (float(pos[i, 0]), float(pos[i, 1])) for i in range(n)}
