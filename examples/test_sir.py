"""TDD: test cases define the desired API, implementation follows."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

# Framework
from abm_framework.core import Agent, Environment, Model

# Domain (SIR)
from examples.sir import SIRRule, sir_agent, count_by_state

# Environment (Grid)
from examples.grid import grid


# ---------------------------------------------------------------------------
# Use case 1: Basic assembly and execution
# ---------------------------------------------------------------------------

def test_basic_assembly_and_run():
    rng = np.random.default_rng(42)

    # Each created independently via closures
    SIRAgent = sir_agent(beta=0.3, gamma=0.1)
    SIRGrid = grid(size=20, periodic=True)

    # Model assembles: Rule(class) + Agent(class) + Env(class)
    model = Model(SIRRule, SIRAgent, SIRGrid, rng=rng)

    # Run with external observer
    observations = model.run(50, observe=count_by_state)

    # Initial state
    assert observations[0].s + observations[0].i + observations[0].r == 400
    assert observations[0].i > 0  # some infected at start

    # Simulation progressed
    assert observations[-1].r > 0  # some recovered by end


# ---------------------------------------------------------------------------
# Use case 2: Conservation law
# ---------------------------------------------------------------------------

def test_population_conservation():
    rng = np.random.default_rng(0)
    SIRAgent = sir_agent(beta=0.3, gamma=0.1)
    SIRGrid = grid(size=10, periodic=True)

    model = Model(SIRRule, SIRAgent, SIRGrid, rng=rng)
    observations = model.run(30, observe=count_by_state)

    population = SIRGrid.size ** 2  # 100
    for o in observations:
        assert o.s + o.i + o.r == population


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
        """Environment without items/nearby — should fail EnvSpec."""
        pass

    with pytest.raises(TypeError):
        Model(SIRRule, SIRAgent, EmptyEnv, rng=rng)


# ---------------------------------------------------------------------------
# Use case 5: Step-by-step execution
# ---------------------------------------------------------------------------

def test_step_and_observe_separately():
    rng = np.random.default_rng(42)
    SIRAgent = sir_agent(beta=0.3, gamma=0.1)
    SIRGrid = grid(size=10, periodic=True)

    model = Model(SIRRule, SIRAgent, SIRGrid, rng=rng)

    before = count_by_state(model.env)
    model.step()
    after = count_by_state(model.env)

    # Population conserved
    assert before.s + before.i + before.r == after.s + after.i + after.r


# ---------------------------------------------------------------------------
# Use case 6: Same grid factory with different rules
# ---------------------------------------------------------------------------

def test_grid_is_rule_agnostic():
    """Grid class produced by grid() knows nothing about SIR."""
    SIRGrid = grid(size=10, periodic=True)

    # Grid class has no SIR-specific attributes
    assert not hasattr(SIRGrid, "beta")
    assert not hasattr(SIRGrid, "gamma")

    # Can be instantiated independently (though Model normally does this)
    # Grid class stores size/periodic but is not yet populated
    assert SIRGrid.size == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
