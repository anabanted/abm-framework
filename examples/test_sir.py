"""TDD: all instances, no closures, no factories."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from abm_framework.core import Agent, Environment, Model
from examples.sir import SIRRule, SIRAgent, count_by_state
from examples.grid import Grid, Cell


# ---------------------------------------------------------------------------
# Use case 1: Basic assembly and execution
# ---------------------------------------------------------------------------

def test_basic_assembly_and_run():
    rng = np.random.default_rng(42)

    rule = SIRRule(
        perception_radius=1.5,
        move_radius=1.5,
        distance_to_weight=lambda d: 1.0 / (d + 0.1),
    )
    agents = (
        [SIRAgent("I", beta=0.3, gamma=0.1) for _ in range(3)]
        + [SIRAgent("S", beta=0.3, gamma=0.1) for _ in range(397)]
    )
    env = Grid(20, Cell, periodic=True)

    model = Model(rule, env, agents, rng=rng)
    observations = model.run(50, observe=count_by_state)

    assert observations[0].s + observations[0].i + observations[0].r == 400
    assert observations[0].i > 0
    assert observations[-1].r > 0


# ---------------------------------------------------------------------------
# Use case 2: Conservation law
# ---------------------------------------------------------------------------

def test_population_conservation():
    rng = np.random.default_rng(0)

    rule = SIRRule(perception_radius=1.5, move_radius=1.5,
                   distance_to_weight=lambda d: 1.0)
    agents = (
        [SIRAgent("I", beta=0.3, gamma=0.1) for _ in range(3)]
        + [SIRAgent("S", beta=0.3, gamma=0.1) for _ in range(97)]
    )
    env = Grid(10, Cell, periodic=True)

    model = Model(rule, env, agents, rng=rng)
    observations = model.run(30, observe=count_by_state)

    for o in observations:
        assert o.s + o.i + o.r == 100


# ---------------------------------------------------------------------------
# Use case 3: Incompatible agent rejection
# ---------------------------------------------------------------------------

def test_reject_incompatible_agent():
    rng = np.random.default_rng(42)

    @dataclass
    class BadAgent(Agent):
        name: str

    rule = SIRRule(perception_radius=1.5, move_radius=1.5,
                   distance_to_weight=lambda d: 1.0)
    agents = [BadAgent("x") for _ in range(25)]
    env = Grid(5, Cell, periodic=False)

    with pytest.raises(TypeError):
        Model(rule, env, agents, rng=rng)


# ---------------------------------------------------------------------------
# Use case 4: Incompatible environment rejection
# ---------------------------------------------------------------------------

def test_reject_incompatible_env():
    rng = np.random.default_rng(42)

    class EmptyEnv(Environment):
        pass

    rule = SIRRule(perception_radius=1.5, move_radius=1.5,
                   distance_to_weight=lambda d: 1.0)
    agents = [SIRAgent("S", beta=0.3, gamma=0.1)]

    with pytest.raises(TypeError):
        Model(rule, EmptyEnv(), agents, rng=rng)


# ---------------------------------------------------------------------------
# Use case 5: Step-by-step
# ---------------------------------------------------------------------------

def test_step_and_observe():
    rng = np.random.default_rng(42)

    rule = SIRRule(perception_radius=1.5, move_radius=1.5,
                   distance_to_weight=lambda d: 1.0)
    agents = (
        [SIRAgent("I", beta=0.3, gamma=0.1) for _ in range(3)]
        + [SIRAgent("S", beta=0.3, gamma=0.1) for _ in range(97)]
    )
    env = Grid(10, Cell, periodic=True)
    model = Model(rule, env, agents, rng=rng)

    before = count_by_state(model.env)
    model.step()
    after = count_by_state(model.env)

    assert before.s + before.i + before.r == after.s + after.i + after.r


# ---------------------------------------------------------------------------
# Use case 6: Individual differences (ABM strength)
# ---------------------------------------------------------------------------

def test_heterogeneous_agents():
    """Agents with different beta/gamma — impossible with ODE."""
    rng = np.random.default_rng(42)

    rule = SIRRule(perception_radius=1.5, move_radius=1.5,
                   distance_to_weight=lambda d: 1.0)
    agents = (
        [SIRAgent("I", beta=0.3, gamma=0.1) for _ in range(3)]
        + [SIRAgent("S", beta=0.1, gamma=0.05) for _ in range(50)]   # cautious
        + [SIRAgent("S", beta=0.5, gamma=0.2) for _ in range(47)]    # active
    )
    env = Grid(10, Cell, periodic=True)
    model = Model(rule, env, agents, rng=rng)

    observations = model.run(30, observe=count_by_state)

    # Population conserved despite heterogeneity
    for o in observations:
        assert o.s + o.i + o.r == 100


# ---------------------------------------------------------------------------
# Use case 7: agents_with_distances
# ---------------------------------------------------------------------------

def test_agents_with_distances():
    rng = np.random.default_rng(42)

    rule = SIRRule(perception_radius=1.5, move_radius=1.5,
                   distance_to_weight=lambda d: 1.0)
    agents = [SIRAgent("S", beta=0.3, gamma=0.1) for _ in range(25)]
    env = Grid(5, Cell, periodic=True)
    model = Model(rule, env, agents, rng=rng)

    loc, _ = next(model.env.items())
    others = model.env.agents_with_distances(loc, max_distance=1.5)
    assert len(others) == 8
    assert all(d <= 1.5 for _, d in others)


# ---------------------------------------------------------------------------
# Use case 8: Dense grid → no movement
# ---------------------------------------------------------------------------

def test_dense_grid_no_movement():
    rng = np.random.default_rng(42)

    rule = SIRRule(perception_radius=1.5, move_radius=1.5,
                   distance_to_weight=lambda d: 1.0)
    agents = [SIRAgent("S", beta=0.3, gamma=0.1) for _ in range(25)]
    env = Grid(5, Cell, periodic=True)
    model = Model(rule, env, agents, rng=rng)

    loc, _ = next(model.env.items())
    assert model.env.reachable(loc, max_distance=1.5) == []


# ---------------------------------------------------------------------------
# Use case 9: choose_move stays when only option
# ---------------------------------------------------------------------------

def test_choose_move_stays():
    rng = np.random.default_rng(42)
    agent = SIRAgent("S", beta=0.3, gamma=0.1)

    assert agent.choose_move([1.0], rng) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
