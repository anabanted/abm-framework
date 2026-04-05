"""SIREnv: composes a Grid with contact_rule + move_rule.

The Grid is a pure spatial structure. SIREnv wraps it with
domain-specific contact and movement rules.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence

import numpy as np

from abm_framework.core import Agent, Environment
from abm_framework.grid import Grid

# Rule signatures
type ContactRule = Callable[[Grid, object], Sequence[tuple[object, float]]]
type MoveRule = Callable[[Grid, object, object, np.random.Generator], None]


class SIREnv(Environment):
    """Wraps a Grid with contact and move rules.

    Grid: pure spatial structure (independent)
    contact_rule: (grid, loc) -> [(agent, intensity)]
    move_rule: (grid, loc, agent, rng) -> None
    """

    def __init__(
        self, grid: Grid, contact_rule: ContactRule, move_rule: MoveRule,
    ) -> None:
        self.grid = grid
        self._contact_rule = contact_rule
        self._move_rule = move_rule

    def place_random(self, agent: Agent, rng: np.random.Generator) -> None:
        self.grid.place_random(agent, rng)

    def __iter__(self) -> Iterator:
        return iter(self.grid)

    def items(self):
        return self.grid.items()

    def contacts(self, location) -> Sequence[tuple[object, float]]:
        """Delegate to composed contact_rule."""
        return self._contact_rule(self.grid, location)

    def try_move(self, location, agent, rng: np.random.Generator) -> None:
        """Delegate to composed move_rule."""
        self._move_rule(self.grid, location, agent, rng)
