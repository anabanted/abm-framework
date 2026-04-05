"""TDD: test cases define the desired API, implementation follows."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from abm_framework.core import Agent, Environment, Model
from examples.sir import SIRRule, sir_agent, count_by_state
from examples.grid import grid


# ---------------------------------------------------------------------------
# Use case 1: Basic assembly and execution
# ---------------------------------------------------------------------------

def test_basic_assembly_and_run():
    rng = np.random.default_rng(42)

    SIRAgent = sir_agent(beta=0.3, gamma=0.1)
    SIRGrid = grid(size=20, periodic=True)

    model = Model(SIRRule, SIRAgent, SIRGrid, rng=rng)
    observations = model.run(50, observe=count_by_state)

    assert observations[0].s + observations[0].i + observations[0].r == 400
    assert observations[0].i > 0
    assert observations[-1].r > 0


# ---------------------------------------------------------------------------
# Use case 2: Conservation law
# ---------------------------------------------------------------------------

def test_population_conservation():
    rng = np.random.default_rng(0)
    SIRAgent = sir_agent(beta=0.3, gamma=0.1)
    SIRGrid = grid(size=10, periodic=True)

    model = Model(SIRRule, SIRAgent, SIRGrid, rng=rng)
    observations = model.run(30, observe=count_by_state)

    for o in observations:
        assert o.s + o.i + o.r == 100


# ---------------------------------------------------------------------------
# Use case 3: Incompatible agent rejection
# ---------------------------------------------------------------------------

def test_reject_incompatible_agent():
    rng = np.random.default_rng(42)

    def bad_agent():
        @dataclass
        class BadAgent(Agent):
            name: str
        return BadAgent

    BadAgent = bad_agent()
    SIRGrid = grid(size=5, periodic=False)

    with pytest.raises(TypeError):
        Model(SIRRule, BadAgent, SIRGrid, rng=rng)


# ---------------------------------------------------------------------------
# Use case 4: Incompatible environment rejection
# ---------------------------------------------------------------------------

def test_reject_incompatible_env():
    rng = np.random.default_rng(42)
    SIRAgent = sir_agent(beta=0.3, gamma=0.1)

    class EmptyEnv(Environment):
        pass

    with pytest.raises(TypeError):
        Model(SIRRule, SIRAgent, EmptyEnv, rng=rng)


# ---------------------------------------------------------------------------
# Use case 5: Step-by-step execution with external observation
# ---------------------------------------------------------------------------

def test_step_and_observe_separately():
    rng = np.random.default_rng(42)
    SIRAgent = sir_agent(beta=0.3, gamma=0.1)
    SIRGrid = grid(size=10, periodic=True)

    model = Model(SIRRule, SIRAgent, SIRGrid, rng=rng)

    before = count_by_state(model.env)
    model.step()
    after = count_by_state(model.env)

    assert before.s + before.i + before.r == after.s + after.i + after.r


# ---------------------------------------------------------------------------
# Use case 6: Grid is rule-agnostic
# ---------------------------------------------------------------------------

def test_grid_is_rule_agnostic():
    SIRGrid = grid(size=10, periodic=True)
    assert SIRGrid.size == 10
    assert not hasattr(SIRGrid, "beta")


# ---------------------------------------------------------------------------
# Use case 7: agents_with_distances returns all agents within max_distance
# ---------------------------------------------------------------------------

def test_agents_with_distances():
    rng = np.random.default_rng(42)
    SIRAgent = sir_agent(beta=0.3, gamma=0.1)
    SIRGrid = grid(size=5, periodic=True)

    model = Model(SIRRule, SIRAgent, SIRGrid, rng=rng)
    env = model.env

    # Pick first cell
    loc, agent = next(env.items())

    # Moore neighborhood: max 8 agents at Chebyshev distance 1
    others = env.agents_with_distances(loc, max_distance=1.5)
    assert len(others) == 8  # 5x5 periodic, all cells filled
    assert all(isinstance(d, float) for _, d in others)
    assert all(d <= 1.5 for _, d in others)

    # Larger radius: more agents
    all_others = env.agents_with_distances(loc, max_distance=100.0)
    assert len(all_others) == 24  # 25 - 1 (self)


# ---------------------------------------------------------------------------
# Use case 8: reachable returns empty for dense grid (no movement)
# ---------------------------------------------------------------------------

def test_dense_grid_no_movement():
    rng = np.random.default_rng(42)
    SIRAgent = sir_agent(beta=0.3, gamma=0.1)
    SIRGrid = grid(size=5, periodic=True)

    model = Model(SIRRule, SIRAgent, SIRGrid, rng=rng)
    env = model.env

    loc, _ = next(env.items())

    # Dense grid: no empty cells, so no reachable destinations
    destinations = env.reachable(loc, max_distance=1.5)
    assert destinations == []


# ---------------------------------------------------------------------------
# Use case 9: choose_move returns None on dense grid
# ---------------------------------------------------------------------------

def test_agent_choose_move_none_on_dense():
    SIRAgent = sir_agent(beta=0.3, gamma=0.1)
    rng = np.random.default_rng(42)
    agent = SIRAgent("S")

    # No destinations available
    result = agent.choose_move([], rng)
    assert result is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
