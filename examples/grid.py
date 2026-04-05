"""Grid environment. Plain class, plain instance."""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import numpy as np

from abm_framework.core import Agent, Environment


class Cell[T]:
    """A grid cell with position and value."""

    __slots__ = ("pos", "value")

    def __init__(self, pos: tuple[int, int], value: T) -> None:
        self.pos = pos
        self.value: T = value

    def __repr__(self) -> str:
        return f"Cell({self.pos}, {self.value!r})"


def _chebyshev(
    a: tuple[int, int], b: tuple[int, int], n: int, periodic: bool,
) -> float:
    dr = abs(a[0] - b[0])
    dc = abs(a[1] - b[1])
    if periodic:
        dr = min(dr, n - dr)
        dc = min(dc, n - dc)
    return float(max(dr, dc))


class Grid(Environment):
    """2D grid environment with Chebyshev distance."""

    def __init__(self, size: int, *, periodic: bool = False) -> None:
        self.size = size
        self.periodic = periodic
        self._cells: list[list[Cell | None]] = [
            [None] * size for _ in range(size)
        ]
        self._agents: list[Cell] = []

    def place_random(self, agent: Agent, rng: np.random.Generator) -> None:
        """Place an agent at a random empty position."""
        empties = [
            (r, c)
            for r in range(self.size)
            for c in range(self.size)
            if self._cells[r][c] is None
        ]
        if not empties:
            raise ValueError("No empty cells available")
        pos = empties[rng.integers(len(empties))]
        cell = Cell(pos, agent)
        self._cells[pos[0]][pos[1]] = cell
        self._agents.append(cell)

    def __iter__(self) -> Iterator:
        return (c.value for c in self._agents)

    def items(self) -> Iterator[tuple[Cell, Agent]]:
        return ((c, c.value) for c in self._agents)

    def agents_with_distances(
        self, location: Cell, max_distance: float,
    ) -> list[tuple[Agent, float]]:
        """All agents within max_distance (excluding self)."""
        return [
            (c.value, d)
            for c in self._agents
            if c is not location
            for d in [_chebyshev(location.pos, c.pos, self.size, self.periodic)]
            if d <= max_distance
        ]

    def reachable(
        self, location: Cell, max_distance: float,
    ) -> list[tuple[tuple[int, int], float]]:
        """Empty cells within max_distance, with distances."""
        return [
            (pos, d)
            for r in range(self.size)
            for c in range(self.size)
            for pos in [(r, c)]
            if self._cells[r][c] is None
            for d in [_chebyshev(location.pos, pos, self.size, self.periodic)]
            if d <= max_distance
        ]

    def move(self, from_loc: Cell, to_pos: tuple[int, int]) -> None:
        """Move agent from cell to empty position."""
        agent = from_loc.value
        self._cells[from_loc.pos[0]][from_loc.pos[1]] = None
        self._agents.remove(from_loc)
        new_cell = Cell(to_pos, agent)
        self._cells[to_pos[0]][to_pos[1]] = new_cell
        self._agents.append(new_cell)
