"""Grid environment and Cell types for the ABM framework."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass

import numpy as np

from .core import Agent, Environment


@dataclass
class Cell[T]:
    """A grid cell that always holds a value."""

    pos: tuple[int, int]
    value: T


@dataclass
class OptionalCell[T]:
    """A grid cell that may be empty (value is None)."""

    pos: tuple[int, int]
    value: T | None = None

    @property
    def is_empty(self) -> bool:
        return self.value is None


def _chebyshev(
    a: tuple[int, int], b: tuple[int, int], n: int, periodic: bool,
) -> float:
    """Chebyshev (L-inf) distance, optionally periodic."""
    dr = abs(a[0] - b[0])
    dc = abs(a[1] - b[1])
    if periodic:
        dr = min(dr, n - dr)
        dc = min(dc, n - dc)
    return float(max(dr, dc))


# Type aliases for rule callables
type ContactRule = Callable[..., Sequence[tuple[object, float]]]
type MoveRule = Callable[..., None]


class Grid[C](Environment):
    """2D grid environment with composable contact and move rules.

    contact_rule: (env, loc) -> [(agent, intensity)]
    move_rule: (env, loc, agent, rng) -> None
    """

    def __init__(
        self,
        size: int,
        cell_factory: Callable[..., C],
        *,
        periodic: bool = False,
        contact_rule: ContactRule,
        move_rule: MoveRule,
    ) -> None:
        self.size = size
        self.periodic = periodic
        self._cell_factory = cell_factory
        self._contact_rule = contact_rule
        self._move_rule = move_rule
        self._cells: list[list[C | None]] = [
            [None] * size for _ in range(size)
        ]
        self._agents: list[C] = []

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
        cell = self._cell_factory(pos, agent)
        self._cells[pos[0]][pos[1]] = cell
        self._agents.append(cell)

    def __iter__(self) -> Iterator:
        return (c.value for c in self._agents)

    def items(self) -> Iterator[tuple[C, object]]:
        return ((c, c.value) for c in self._agents)

    def contacts(self, location: C) -> Sequence[tuple[object, float]]:
        """Get contacts for an agent using the composed contact_rule."""
        return self._contact_rule(self, location)

    def try_move(self, location: C, agent: object, rng: np.random.Generator) -> None:
        """Attempt to move an agent using the composed move_rule."""
        self._move_rule(self, location, agent, rng)

    def move(self, from_loc: C, to_pos: tuple[int, int]) -> None:
        """Execute a move (used by move_rules)."""
        agent = from_loc.value
        self._cells[from_loc.pos[0]][from_loc.pos[1]] = None
        self._agents.remove(from_loc)
        new_cell = self._cell_factory(to_pos, agent)
        self._cells[to_pos[0]][to_pos[1]] = new_cell
        self._agents.append(new_cell)
