"""Animation helpers for all environment types."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.patches import Patch
from matplotlib.animation import FuncAnimation
from numpy.typing import NDArray

from .core import Environment, Model


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------


def grid_to_color_array(
    env,
    state_to_index: dict[str, int],
    *,
    empty_index: int = 0,
) -> NDArray[np.int_]:
    """Convert a Grid environment to a 2D int array for visualization."""
    return np.array([
        [state_to_index.get(cell.value.state, empty_index) if cell is not None
         else empty_index
         for cell in row]
        for row in env._cells
    ])


def record_simulation[O](
    model: Model,
    n_steps: int,
    observe: Callable[[Environment], O],
    snapshot: Callable[[Environment], object],
    *,
    progress: bool = True,
) -> tuple[list, list[O]]:
    """Run simulation and record snapshots + observations at each step."""
    try:
        from tqdm.auto import trange
        step_iter = trange(n_steps, desc="Simulating", disable=not progress)
    except ImportError:
        step_iter = range(n_steps)
    snapshots = [snapshot(model.env)]
    observations = [observe(model.env)]
    for _ in step_iter:
        model.step()
        snapshots.append(snapshot(model.env))
        observations.append(observe(model.env))
    return snapshots, observations


def _make_anim(fig, update, n_steps, interval=80):
    """Create FuncAnimation with standard frame skipping."""
    frame_indices = list(range(0, n_steps + 1, 2))
    anim = FuncAnimation(fig, lambda fi: update(frame_indices[fi]),
                         frames=len(frame_indices), interval=interval, blit=False)
    plt.close(fig)
    return anim


# ---------------------------------------------------------------------------
# animate_dense_grid — Grid[Cell[A]], all cells filled
# ---------------------------------------------------------------------------


def animate_dense_grid[O](
    grid_snapshots: list[NDArray],
    color_map: NDArray,
    observations: list[O],
    n_steps: int,
    title: str,
    legend_entries: Sequence[tuple[str, str]],
    setup_right: Callable[[Axes, list[O], int], None],
    update_right: Callable[[Axes, list[O], int], str],
) -> FuncAnimation:
    """Animate a dense grid (no empty cells)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(title, fontsize=12, fontweight="bold")

    im = ax1.imshow(color_map[grid_snapshots[0]], interpolation="nearest")
    ax1.set_xticks([]); ax1.set_yticks([])
    title1 = ax1.set_title("Step 0")
    ax1.legend(handles=[Patch(facecolor=c, label=l) for c, l in legend_entries],
               loc="upper right", fontsize=9)
    setup_right(ax2, observations, n_steps)

    def update(t):
        im.set_data(color_map[grid_snapshots[t]])
        title1.set_text(f"Step {t} \u2014 {update_right(ax2, observations, t)}")

    return _make_anim(fig, update, n_steps)


# ---------------------------------------------------------------------------
# animate_sparse_grid — Grid[OptionalCell[A]], some cells empty
# ---------------------------------------------------------------------------


def animate_sparse_grid[O](
    grid_snapshots: list[NDArray],
    color_map: NDArray,
    observations: list[O],
    n_steps: int,
    title: str,
    legend_entries: Sequence[tuple[str, str]],
    setup_right: Callable[[Axes, list[O], int], None],
    update_right: Callable[[Axes, list[O], int], str],
) -> FuncAnimation:
    """Animate a sparse grid (empty cells shown as distinct color)."""
    # Same rendering as dense — color_map[0] handles empty cells
    return animate_dense_grid(
        grid_snapshots, color_map, observations, n_steps,
        title, legend_entries, setup_right, update_right,
    )


# ---------------------------------------------------------------------------
# animate_graph — Graph with edges and node positions
# ---------------------------------------------------------------------------


def animate_graph[O](
    snapshots: list[list[tuple[tuple[float, float], str]]],
    observations: list[O],
    n_steps: int,
    title: str,
    state_colors: dict[str, str],
    edges: list[tuple[int, int]],
    node_pos: dict[int, tuple[float, float]],
    setup_right: Callable[[Axes, list[O], int], None],
    update_right: Callable[[Axes, list[O], int], str],
) -> FuncAnimation:
    """Animate agents on a graph with edges drawn."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(title, fontsize=12, fontweight="bold")
    setup_right(ax2, observations, n_steps)

    def update(t):
        ax1.cla()
        for i, j in edges:
            ax1.plot([node_pos[i][0], node_pos[j][0]],
                     [node_pos[i][1], node_pos[j][1]],
                     color="lightgrey", lw=0.5, zorder=1)
        for (x, y), state in snapshots[t]:
            ax1.scatter(x, y, c=state_colors[state], s=20, alpha=0.8, zorder=2)
        ax1.set_aspect("equal")
        ax1.set_xticks([]); ax1.set_yticks([])
        ax1.set_title(f"Step {t} \u2014 {update_right(ax2, observations, t)}")

    return _make_anim(fig, update, n_steps, interval=100)


# ---------------------------------------------------------------------------
# animate_freespace — continuous 2D with moving particles
# ---------------------------------------------------------------------------


def animate_freespace[O](
    snapshots: list[list[tuple[tuple[float, float], str]]],
    observations: list[O],
    n_steps: int,
    title: str,
    state_colors: dict[str, str],
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    setup_right: Callable[[Axes, list[O], int], None],
    update_right: Callable[[Axes, list[O], int], str],
) -> FuncAnimation:
    """Animate agents in continuous 2D space."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(title, fontsize=12, fontweight="bold")
    setup_right(ax2, observations, n_steps)

    def update(t):
        ax1.cla()
        for (x, y), state in snapshots[t]:
            ax1.scatter(x, y, c=state_colors[state], s=12, alpha=0.7)
        ax1.set_xlim(*xlim); ax1.set_ylim(*ylim)
        ax1.set_aspect("equal")
        ax1.set_xticks([]); ax1.set_yticks([])
        ax1.set_title(f"Step {t} \u2014 {update_right(ax2, observations, t)}")

    return _make_anim(fig, update, n_steps, interval=100)


# Backward compatibility
animate_grid = animate_dense_grid
