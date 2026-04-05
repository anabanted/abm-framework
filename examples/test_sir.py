"""TDD: Grid is pure spatial. SIREnv composes Grid + rules."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from abm_framework.core import Agent, Environment, Model
from abm_framework.grid import Grid, Cell, OptionalCell
from examples.sir import SIRRule, SIRAgent, count_by_state
from examples.sir_env import SIREnv
from examples.grid_rules import moore_contacts, no_move, adjacent_move


# ---------------------------------------------------------------------------
# Use case 1: Dense grid, no movement
# ---------------------------------------------------------------------------

def test_dense_grid():
    rng = np.random.default_rng(42)

    # Grid: pure spatial (no rules)
    grid = Grid(20, Cell, periodic=True)

    # SIREnv: composes Grid + contact/move rules
    env = SIREnv(grid, contact_rule=moore_contacts, move_rule=no_move)

    rule = SIRRule()
    agents = (
        [SIRAgent("I", beta=0.3, gamma=0.1) for _ in range(3)]
        + [SIRAgent("S", beta=0.3, gamma=0.1) for _ in range(397)]
    )
    model = Model(rule, env, agents, rng=rng)
    observations = model.run(50, observe=count_by_state)

    assert observations[0].s + observations[0].i + observations[0].r == 400
    assert observations[0].i > 0
    assert observations[-1].r > 0


# ---------------------------------------------------------------------------
# Use case 2: Conservation
# ---------------------------------------------------------------------------

def test_population_conservation():
    rng = np.random.default_rng(0)

    grid = Grid(10, Cell, periodic=True)
    env = SIREnv(grid, contact_rule=moore_contacts, move_rule=no_move)

    rule = SIRRule()
    agents = (
        [SIRAgent("I", beta=0.3, gamma=0.1) for _ in range(3)]
        + [SIRAgent("S", beta=0.3, gamma=0.1) for _ in range(97)]
    )
    model = Model(rule, env, agents, rng=rng)
    observations = model.run(30, observe=count_by_state)

    for o in observations:
        assert o.s + o.i + o.r == 100


# ---------------------------------------------------------------------------
# Use case 3: Sparse grid with movement
# ---------------------------------------------------------------------------

def test_sparse_grid_with_movement():
    rng = np.random.default_rng(42)

    # Same Grid class, OptionalCell, different move rule
    grid = Grid(20, OptionalCell, periodic=True)
    env = SIREnv(grid, contact_rule=moore_contacts, move_rule=adjacent_move)

    rule = SIRRule()
    agents = (
        [SIRAgent("I", beta=0.3, gamma=0.1) for _ in range(3)]
        + [SIRAgent("S", beta=0.3, gamma=0.1) for _ in range(267)]
    )
    model = Model(rule, env, agents, rng=rng)
    observations = model.run(50, observe=count_by_state)

    for o in observations:
        assert o.s + o.i + o.r == 270


# ---------------------------------------------------------------------------
# Use case 4: Contacts return intensity
# ---------------------------------------------------------------------------

def test_contacts_returns_intensity():
    rng = np.random.default_rng(42)

    grid = Grid(5, Cell, periodic=True)
    env = SIREnv(grid, contact_rule=moore_contacts, move_rule=no_move)

    rule = SIRRule()
    agents = [SIRAgent("S", beta=0.3, gamma=0.1) for _ in range(25)]
    model = Model(rule, env, agents, rng=rng)

    loc, _ = next(model.env.items())
    contacts = model.env.contacts(loc)
    assert len(contacts) == 8
    assert all(isinstance(intensity, float) for _, intensity in contacts)


# ---------------------------------------------------------------------------
# Use case 5: Grid is independent (no rules attached)
# ---------------------------------------------------------------------------

def test_grid_is_independent():
    grid = Grid(10, Cell, periodic=True)
    assert grid.size == 10
    assert not hasattr(grid, "_contact_rule")
    assert not hasattr(grid, "_move_rule")


# ---------------------------------------------------------------------------
# Use case 6: Heterogeneous agents
# ---------------------------------------------------------------------------

def test_heterogeneous_agents():
    rng = np.random.default_rng(42)

    grid = Grid(10, Cell, periodic=True)
    env = SIREnv(grid, contact_rule=moore_contacts, move_rule=no_move)

    rule = SIRRule()
    agents = (
        [SIRAgent("I", beta=0.3, gamma=0.1) for _ in range(3)]
        + [SIRAgent("S", beta=0.1, gamma=0.05) for _ in range(50)]
        + [SIRAgent("S", beta=0.5, gamma=0.2) for _ in range(47)]
    )
    model = Model(rule, env, agents, rng=rng)
    observations = model.run(30, observe=count_by_state)

    for o in observations:
        assert o.s + o.i + o.r == 100


# ---------------------------------------------------------------------------
# Use case 7: Incompatible agent
# ---------------------------------------------------------------------------

def test_reject_incompatible_agent():
    rng = np.random.default_rng(42)

    @dataclass
    class BadAgent(Agent):
        name: str

    grid = Grid(5, Cell, periodic=False)
    env = SIREnv(grid, contact_rule=moore_contacts, move_rule=no_move)

    rule = SIRRule()

    with pytest.raises(TypeError):
        Model(rule, env, [BadAgent("x") for _ in range(25)], rng=rng)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
