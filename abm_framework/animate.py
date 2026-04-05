"""Grid animation helpers for the ABM framework."""

from __future__ import annotations

from collections.abc import Callable
from typing import Sequence

import numpy as np
import matplotlib.pyplot as plt
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
    """Convert a Grid environment to a 2D int array for visualization.

    Maps agent states to integer indices via state_to_index.
    Empty cells (None) map to empty_index.
    """
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


def animate_grid(
    grid_snapshots: list[NDArray],
    color_map: NDArray,
    curves: dict[str, tuple[list[float], str]],
    n_steps: int,
    y_max: int,
    title: str,
    legend_entries: Sequence[tuple[str, str]],
) -> FuncAnimation:
    """Create a grid + curves animation.

    Parameters
    ----------
    grid_snapshots : list of 2D int arrays (indexed into color_map)
    color_map : (N, 3) array mapping indices to RGB
    curves : {"label": (values_list, color_string), ...}
    n_steps : number of simulation steps
    y_max : y-axis maximum for curves
    title : figure title
    legend_entries : [(color, label), ...] for grid legend
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

    t_all = np.arange(len(grid_snapshots))
    lines = {}
    for label, (values, color) in curves.items():
        ax2.plot(t_all, values, color=color, alpha=0.15)
        line, = ax2.plot([], [], color=color, lw=2, label=label)
        lines[label] = (line, values)
    vline = ax2.axvline(x=0, color="black", ls="--", lw=1, alpha=0.7)
    ax2.set(xlabel="Time Step", ylabel="Number of Agents",
            xlim=(0, n_steps), ylim=(0, y_max))
    ax2.legend(loc="center right")

    frame_indices = list(range(0, n_steps + 1, 2))

    def update(frame_idx: int):
        t = frame_indices[frame_idx]
        im.set_data(color_map[grid_snapshots[t]])
        parts = "  ".join(f"{l}={int(v[t])}" for l, (_, v) in curves.items())
        title1.set_text(f"Step {t} \u2014 {parts}")
        for line, values in lines.values():
            line.set_data(t_all[:t+1], values[:t+1])
        vline.set_xdata([t, t])

    anim = FuncAnimation(fig, update, frames=len(frame_indices),
                         interval=80, blit=False)
    plt.close(fig)
    return anim
