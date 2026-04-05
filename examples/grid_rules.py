"""Contact and movement rules for Grid environments."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from abm_framework.grid import Grid, Cell, _chebyshev


# ---------------------------------------------------------------------------
# Contact rules: (env, loc) -> Sequence[tuple[Agent, float]]
# ---------------------------------------------------------------------------


def moore_contacts(env: Grid, loc) -> list[tuple[object, float]]:
    """Moore neighborhood (8-adjacent), intensity 1.0 for all."""
    return [
        (cell.value, 1.0)
        for cell in env._agents
        if cell is not loc
        and _chebyshev(loc.pos, cell.pos, env.size, env.periodic) <= 1.5
    ]


# ---------------------------------------------------------------------------
# Move rules: (env, loc, agent, rng) -> None
# ---------------------------------------------------------------------------


def no_move(env: Grid, loc, agent, rng: np.random.Generator) -> None:
    """No movement — agent stays in place."""


def adjacent_move(env: Grid, loc, agent, rng: np.random.Generator) -> None:
    """Move to a random adjacent empty cell (Chebyshev distance <= 1)."""
    empties = [
        (r, c)
        for r in range(env.size)
        for c in range(env.size)
        if env._cells[r][c] is None
        and _chebyshev(loc.pos, (r, c), env.size, env.periodic) <= 1.0
    ]
    if empties:
        dest = empties[rng.integers(len(empties))]
        env.move(loc, dest)
