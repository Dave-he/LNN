"""Synthetic Pathfinder data generator (LRA-style long-range curve tracing).

Pathfinder (LRA) is a binary classification task: given a 32x32 grayscale
image with two endpoint markers and 0-2 curved paths, decide whether the
two endpoints are connected by any path. The challenge is that the model
must integrate information across the *full* 1024-length pixel sequence
(endpoints can be on opposite corners), making it a standard LRA long-range
benchmark.

This module provides a *synthetic* generator — no external download required.
Generation algorithm (deterministic given ``seed``):

1. Pick two endpoint cells at random positions on a 32x32 grid.
2. With probability 0.5, draw a connecting curve (otherwise no curve).
   - A "curve" is a piecewise-linear polyline through 2-4 control points
     including the two endpoints.
   - It is rasterized onto the 32x32 grid by stepping along the polyline
     and marking each grid cell the line passes through as "on path".
3. Output:
   - seq: [1024] float tensor in [0, 1] — flattened image (background 0.2,
     on-path 0.6, endpoint 1.0)
   - label: 0/1 long tensor — endpoints connected?

Class balance: 50/50 (uniform).

The synthetic version is simpler than the real Pathfinder (which has
multiple curve segments and varying complexity), but preserves the
core long-range signal: the model must integrate information across
the full 1024 sequence to decide connectivity.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass
class PathfinderConfig:
    grid_size: int = 32
    n_curves_max: int = 2       # max number of polyline curves per image
    n_waypoints_min: int = 2    # min polyline control points (incl. endpoints)
    n_waypoints_max: int = 4
    bg_value: float = 0.2
    path_value: float = 0.6
    endpoint_value: float = 1.0
    endpoint_radius: int = 1    # 1 cell = 3x3 filled marker


def _draw_line(grid: torch.Tensor, p0: tuple[int, int], p1: tuple[int, int], value: float) -> None:
    """Rasterize a line from p0 to p1 onto ``grid`` (in-place) using Bresenham."""
    H, W = grid.shape
    x0, y0 = p0
    x1, y1 = p1
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    while True:
        if 0 <= x0 < W and 0 <= y0 < H:
            # Existing-path max: don't overwrite endpoint markers (1.0).
            if grid[y0, x0].item() < value:
                grid[y0, x0] = value
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy


def _draw_endpoint_marker(grid: torch.Tensor, cx: int, cy: int, radius: int, value: float) -> None:
    """Draw a small filled circle at (cx, cy)."""
    H, W = grid.shape
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx * dx + dy * dy <= radius * radius:
                x, y = cx + dx, cy + dy
                if 0 <= x < W and 0 <= y < H:
                    grid[y, x] = value


def _generate_one(cfg: PathfinderConfig, rng: torch.Generator) -> tuple[torch.Tensor, int]:
    """Generate one synthetic Pathfinder example (returns seq [1024] and label)."""
    H = cfg.grid_size
    W = cfg.grid_size

    # Pick two endpoint cells, ensuring they're not adjacent (so connectivity is non-trivial).
    def _rand_cell() -> tuple[int, int]:
        x = int(torch.randint(2, W - 2, (1,), generator=rng).item())
        y = int(torch.randint(2, H - 2, (1,), generator=rng).item())
        return x, y

    p0 = _rand_cell()
    while True:
        p1 = _rand_cell()
        d = math.hypot(p0[0] - p1[0], p0[1] - p1[1])
        if d >= 16.0:  # endpoints at least 16 cells apart (half the diagonal)
            break

    # With 0.5 probability, draw a connecting curve; otherwise no curve.
    connected = bool(float(torch.rand(1, generator=rng).item()) < 0.5)

    grid = torch.full((H, W), cfg.bg_value)

    if connected:
        # Build a polyline through 2-4 control points including the two endpoints.
        n_wp = int(torch.randint(cfg.n_waypoints_min, cfg.n_waypoints_max + 1, (1,), generator=rng).item())
        waypoints = [p0]
        for _ in range(n_wp - 2):
            waypoints.append(_rand_cell())
        waypoints.append(p1)
        # Draw segments
        for a, b in zip(waypoints[:-1], waypoints[1:]):
            _draw_line(grid, a, b, cfg.path_value)

    # Mark endpoints last (so they overwrite the path)
    _draw_endpoint_marker(grid, p0[0], p0[1], cfg.endpoint_radius, cfg.endpoint_value)
    _draw_endpoint_marker(grid, p1[0], p1[1], cfg.endpoint_radius, cfg.endpoint_value)

    seq = grid.view(-1)  # [H*W] = [1024] for H=W=32
    label = 1 if connected else 0
    return seq, label


def generate_pathfinder(
    n_samples: int,
    cfg: PathfinderConfig | None = None,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate ``n_samples`` synthetic Pathfinder examples.

    Returns:
        seqs:   [N, H*W] float tensor in [0, 1]
        labels: [N] long tensor of 0/1
    """
    if cfg is None:
        cfg = PathfinderConfig()
    rng = torch.Generator()
    rng.manual_seed(seed)
    H, W = cfg.grid_size, cfg.grid_size
    seqs = torch.zeros(n_samples, H * W)
    labels = torch.zeros(n_samples, dtype=torch.long)
    for i in range(n_samples):
        seq, lab = _generate_one(cfg, rng)
        seqs[i] = seq
        labels[i] = lab
    return seqs, labels


__all__ = ["PathfinderConfig", "generate_pathfinder"]
