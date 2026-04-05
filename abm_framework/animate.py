"""Grid animation helpers for the ABM framework."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.patches import Patch
from matplotlib.animation import FuncAnimation
from numpy.typing import NDArray

from .core import Environment, Model


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
    snapshot: Callable[[Environment], NDArray],
) -> tuple[list[NDArray], list[O]]:
    """Run simulation and record grid snapshots + observations at each step."""
    snapshots = [snapshot(model.env)]
    observations = [observe(model.env)]
    for _ in range(n_steps):
        model.step()
        snapshots.append(snapshot(model.env))
        observations.append(observe(model.env))
    return snapshots, observations


def animate_grid[O](
    grid_snapshots: list[NDArray],
    color_map: NDArray,
    observations: list[O],
    n_steps: int,
    title: str,
    legend_entries: Sequence[tuple[str, str]],
    setup_right: Callable[[Axes, list[O], int], None],
    update_right: Callable[[Axes, list[O], int], str],
) -> FuncAnimation:
    """Create a grid animation with injectable right panel.

    Parameters
    ----------
    grid_snapshots : list of 2D int arrays (indexed into color_map)
    color_map : (N, 3) array mapping indices to RGB
    observations : list of observation objects (one per step)
    n_steps : number of simulation steps
    title : figure title
    legend_entries : [(color, label), ...] for grid legend
    setup_right : (ax, observations, n_steps) -> None
        Called once to initialize the right panel
    update_right : (ax, observations, t) -> str
        Called each frame. Returns subtitle text for the grid panel.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(title, fontsize=12, fontweight="bold")

    im = ax1.imshow(color_map[grid_snapshots[0]], interpolation="nearest")
    ax1.set_xticks([]); ax1.set_yticks([])
    title1 = ax1.set_title("Step 0")
    ax1.legend(
        handles=[Patch(facecolor=c, label=l) for c, l in legend_entries],
        loc="upper right", fontsize=9,
    )

    setup_right(ax2, observations, n_steps)

    frame_indices = list(range(0, n_steps + 1, 2))

    def update(frame_idx: int):
        t = frame_indices[frame_idx]
        im.set_data(color_map[grid_snapshots[t]])
        subtitle = update_right(ax2, observations, t)
        title1.set_text(f"Step {t} \u2014 {subtitle}")

    anim = FuncAnimation(fig, update, frames=len(frame_indices),
                         interval=80, blit=False)
    plt.close(fig)
    return anim
