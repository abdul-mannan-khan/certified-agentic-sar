"""Reachable waypoint sets under no-fly-zone constraints.

A_i(k) is the set of grid cells that UAV i can reach within one tactical
epoch (duration dt_epoch at cruise speed v) along a path that stays outside
every no-fly zone (NFZ) inflated by a safety margin. Encoding travel here is
what makes the per-epoch choice a partition-matroid problem: each UAV picks
one element of its own reachable set.
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass

import numpy as np


@dataclass
class NFZ:
    """Circular no-fly zone moving with constant velocity (bounded speed)."""
    x: float
    y: float
    r: float
    vx: float = 0.0
    vy: float = 0.0

    def at(self, t: float) -> "NFZ":
        return NFZ(self.x + self.vx * t, self.y + self.vy * t, self.r, self.vx, self.vy)

    @property
    def speed(self) -> float:
        return float(np.hypot(self.vx, self.vy))


def blocked_mask(centres: np.ndarray, nfzs, margin: float, horizon: float = 0.0) -> np.ndarray:
    """Cells whose centre lies within r + margin + v_max * horizon of any NFZ.

    The extra v * horizon term is the constraint tightening of Theorem 5: a
    cell that is free now stays free for the whole epoch if the NFZ cannot
    move further than v * horizon in that time.
    """
    blocked = np.zeros(len(centres), dtype=bool)
    for z in nfzs:
        d = np.hypot(centres[:, 0] - z.x, centres[:, 1] - z.y)
        blocked |= d <= z.r + margin + z.speed * horizon
    return blocked


def reachable_cells(start: int, n_rows: int, n_cols: int, cell_size: float,
                    blocked: np.ndarray, max_dist: float) -> dict[int, float]:
    """Dijkstra over the 8-connected free grid; returns {cell: path length}.

    The start cell is always included (hovering is always feasible), even if a
    moving NFZ has just reached it, so that the action set is never empty.
    """
    dist = {start: 0.0}
    heap = [(0.0, start)]
    steps = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist.get(u, np.inf):
            continue
        r, c = divmod(u, n_cols)
        for dr, dc in steps:
            rr, cc = r + dr, c + dc
            if not (0 <= rr < n_rows and 0 <= cc < n_cols):
                continue
            v = rr * n_cols + cc
            if blocked[v]:
                continue
            # Disallow cutting corners between two blocked cells.
            if dr and dc and blocked[r * n_cols + cc] and blocked[rr * n_cols + c]:
                continue
            nd = d + cell_size * (1.4142135623730951 if dr and dc else 1.0)
            if nd <= max_dist + 1e-9 and nd < dist.get(v, np.inf):
                dist[v] = nd
                heapq.heappush(heap, (nd, v))
    return dist


def grid_path(a: int, b: int, n_rows: int, n_cols: int, blocked: np.ndarray) -> list[int]:
    """Shortest 8-connected free path of cells from a to b (no corner cutting).
    Cell a may be blocked (a UAV caught by a moving NFZ must be able to leave).
    Returns [a] if b is unreachable."""
    dist, prev, h = {a: 0.0}, {}, [(0.0, a)]
    while h:
        d, u = heapq.heappop(h)
        if u == b:
            break
        if d > dist[u]:
            continue
        r, c = divmod(u, n_cols)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr, cc = r + dr, c + dc
                if (dr or dc) and 0 <= rr < n_rows and 0 <= cc < n_cols:
                    v = rr * n_cols + cc
                    if blocked[v] or (dr and dc and (blocked[r * n_cols + cc] or blocked[rr * n_cols + c])):
                        continue
                    nd = d + (1.4142135623730951 if dr and dc else 1.0)
                    if nd < dist.get(v, np.inf):
                        dist[v], prev[v] = nd, u
                        heapq.heappush(h, (nd, v))
    if b not in dist:
        return [a]
    path = [b]
    while path[-1] != a:
        path.append(prev[path[-1]])
    return path[::-1]


def free_distances(src: int, n_rows: int, n_cols: int, cell_size: float, blocked: np.ndarray) -> np.ndarray:
    """Free-space path length (m) from src to every cell (inf if unreachable)."""
    d = reachable_cells(src, n_rows, n_cols, cell_size, blocked, np.inf)
    out = np.full(n_rows * n_cols, np.inf)
    for k, v in d.items():
        out[k] = v
    return out
