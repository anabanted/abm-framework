"""TDD: contact_rule + move_rule composed into env, called via Rule→Agent→Env."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from abm_framework.core import Agent, Environment, Model
from abm_framework.grid import Grid, Cell, OptionalCell
from examples.sir import SIRRule, SIRAgent, count_by_state


# ---------------------------------------------------------------------------
# Use case 1: Dense grid, no movement
# ---------------------------------------------------------------------------

def test_dense_grid():
    rng = np.random.default_rng(42)

    rule = SIRRule()
    agents = (
        [SIRAgent("I", beta=0.3, gamma=0.1) for _ in range(3)]
        + [SIRAgent("S", beta=0.3, gamma=0.1) for _ in range(397)]
    )
    from examples.grid_rules import moore_contacts, no_move
    env = Grid(20, Cell, periodic=True,
               contact_rule=moore_contacts, move_rule=no_move)

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
    from examples.grid_rules import moore_contacts, no_move

    rule = SIRRule()
    agents = (
        [SIRAgent("I", beta=0.3, gamma=0.1) for _ in range(3)]
        + [SIRAgent("S", beta=0.3, gamma=0.1) for _ in range(97)]
    )
    env = Grid(10, Cell, periodic=True,
               contact_rule=moore_contacts, move_rule=no_move)

    model = Model(rule, env, agents, rng=rng)
    observations = model.run(30, observe=count_by_state)

    for o in observations:
        assert o.s + o.i + o.r == 100


# ---------------------------------------------------------------------------
# Use case 3: Sparse grid with movement
# ---------------------------------------------------------------------------

def test_sparse_grid_with_movement():
    rng = np.random.default_rng(42)
    from examples.grid_rules import moore_contacts, adjacent_move

    rule = SIRRule()
    agents = (
        [SIRAgent("I", beta=0.3, gamma=0.1) for _ in range(3)]
        + [SIRAgent("S", beta=0.3, gamma=0.1) for _ in range(267)]
    )
    env = Grid(20, OptionalCell, periodic=True,
               contact_rule=moore_contacts, move_rule=adjacent_move)

    model = Model(rule, env, agents, rng=rng)
    observations = model.run(50, observe=count_by_state)

    for o in observations:
        assert o.s + o.i + o.r == 270


# ---------------------------------------------------------------------------
# Use case 4: contacts returns (agent, intensity) pairs
# ---------------------------------------------------------------------------

def test_contacts_returns_intensity():
    rng = np.random.default_rng(42)
    from examples.grid_rules import moore_contacts, no_move

    rule = SIRRule()
    agents = [SIRAgent("S", beta=0.3, gamma=0.1) for _ in range(25)]
    env = Grid(5, Cell, periodic=True,
               contact_rule=moore_contacts, move_rule=no_move)
    model = Model(rule, env, agents, rng=rng)

    loc, _ = next(model.env.items())
    contacts = model.env.contacts(loc)
    assert len(contacts) == 8  # Moore neighborhood
    assert all(isinstance(intensity, float) for _, intensity in contacts)


# ---------------------------------------------------------------------------
# Use case 5: Dense grid has no movement (reachable empty)
# ---------------------------------------------------------------------------

def test_dense_no_reachable():
    rng = np.random.default_rng(42)
    from examples.grid_rules import moore_contacts, adjacent_move

    rule = SIRRule()
    agents = [SIRAgent("S", beta=0.3, gamma=0.1) for _ in range(25)]
    # Even with adjacent_move, dense grid can't move
    env = Grid(5, Cell, periodic=True,
               contact_rule=moore_contacts, move_rule=adjacent_move)
    model = Model(rule, env, agents, rng=rng)

    # After step, nothing moves (no empty cells)
    before = [a.state for a in model.env]
    model.step()
    # Population conserved
    assert sum(1 for a in model.env) == 25


# ---------------------------------------------------------------------------
# Use case 6: Heterogeneous agents
# ---------------------------------------------------------------------------

def test_heterogeneous_agents():
    rng = np.random.default_rng(42)
    from examples.grid_rules import moore_contacts, no_move

    rule = SIRRule()
    agents = (
        [SIRAgent("I", beta=0.3, gamma=0.1) for _ in range(3)]
        + [SIRAgent("S", beta=0.1, gamma=0.05) for _ in range(50)]
        + [SIRAgent("S", beta=0.5, gamma=0.2) for _ in range(47)]
    )
    env = Grid(10, Cell, periodic=True,
               contact_rule=moore_contacts, move_rule=no_move)
    model = Model(rule, env, agents, rng=rng)
    observations = model.run(30, observe=count_by_state)

    for o in observations:
        assert o.s + o.i + o.r == 100


# ---------------------------------------------------------------------------
# Use case 7: Incompatible agent
# ---------------------------------------------------------------------------

def test_reject_incompatible_agent():
    rng = np.random.default_rng(42)
    from examples.grid_rules import moore_contacts, no_move

    @dataclass
    class BadAgent(Agent):
        name: str

    rule = SIRRule()
    env = Grid(5, Cell, periodic=False,
               contact_rule=moore_contacts, move_rule=no_move)

    with pytest.raises(TypeError):
        Model(rule, env, [BadAgent("x") for _ in range(25)], rng=rng)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
