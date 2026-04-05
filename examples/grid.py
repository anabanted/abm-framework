"""Grid environment factory.

grid(size, periodic) returns a Grid CLASS (not instance).
The class is instantiated by Model.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import numpy as np

from abm_framework.core import Agent, Environment


class Cell[T]:
    """A grid cell that always holds a value, aware of its position."""

    __slots__ = ("pos", "value")

    def __init__(self, pos: tuple[int, int], value: T) -> None:
        self.pos = pos
        self.value: T = value

    def __repr__(self) -> str:
        return f"Cell({self.pos}, {self.value!r})"


def _chebyshev(
    a: tuple[int, int], b: tuple[int, int], n: int, periodic: bool,
) -> float:
    """Chebyshev (L∞) distance, optionally periodic."""
    dr = abs(a[0] - b[0])
    dc = abs(a[1] - b[1])
    if periodic:
        dr = min(dr, n - dr)
        dc = min(dc, n - dc)
    return float(max(dr, dc))


def grid(size: int, *, periodic: bool = False) -> type[Environment]:
    """Factory: returns a Grid class with size/periodic baked in."""

    _grid_size = size
    _grid_periodic = periodic

    class Grid(Environment):
        """2D grid environment with Chebyshev distance."""

        size = _grid_size
        periodic = _grid_periodic

        def __init__(self) -> None:
            self._cells: list[list[Cell | None]] = [
                [None for _ in range(self.size)]
                for _ in range(self.size)
            ]
            self._agents: list[Cell] = []

        def place(self, agent: Agent, pos: tuple[int, int]) -> None:
            """Place an agent at a specific grid position."""
            cell = Cell(pos, agent)
            self._cells[pos[0]][pos[1]] = cell
            self._agents.append(cell)

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
            self.place(agent, pos)

        def __iter__(self) -> Iterator:
            """Iterate over all agents."""
            return (cell.value for cell in self._agents)

        def items(self) -> Iterator[tuple[Cell, Agent]]:
            """Iterate over (cell, agent) pairs."""
            return ((cell, cell.value) for cell in self._agents)

        def nearby(
            self, location: Cell, radius: float,
        ) -> list[tuple[Agent, float]]:
            """Agents within Chebyshev distance <= radius (excluding self)."""
            return [
                (cell.value, dist)
                for cell in self._agents
                if cell is not location
                for dist in [_chebyshev(
                    location.pos, cell.pos, self.size, self.periodic,
                )]
                if dist <= radius
            ]

    return Grid
