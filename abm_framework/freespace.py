"""FreeSpace: continuous 2D space with periodic boundaries."""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from .core import Agent, Environment


@dataclass
class Particle[T]:
    """An agent with a continuous position."""

    pos: tuple[float, float]
    value: T


class FreeSpace(Environment):
    """Continuous 2D space. Agents are Particles with (x, y) positions."""

    def __init__(
        self, width: float, height: float, *, periodic: bool = True,
    ) -> None:
        self.width = width
        self.height = height
        self.periodic = periodic
        self._particles: list[Particle] = []

    def place_random(self, agent: Agent, rng: np.random.Generator) -> None:
        pos = (float(rng.uniform(0, self.width)),
               float(rng.uniform(0, self.height)))
        self._particles.append(Particle(pos, agent))

    def __iter__(self) -> Iterator:
        return (p.value for p in self._particles)

    def items(self) -> Iterator[tuple[Particle, Agent]]:
        return ((p, p.value) for p in self._particles)

    def distance(self, a: tuple[float, float], b: tuple[float, float]) -> float:
        """Euclidean distance, optionally periodic."""
        dx = abs(a[0] - b[0])
        dy = abs(a[1] - b[1])
        if self.periodic:
            dx = min(dx, self.width - dx)
            dy = min(dy, self.height - dy)
        return math.sqrt(dx * dx + dy * dy)

    def move_particle(self, particle: Particle, new_pos: tuple[float, float]) -> None:
        """Move a particle, wrapping if periodic."""
        x, y = new_pos
        if self.periodic:
            x = x % self.width
            y = y % self.height
        particle.pos = (x, y)
