"""Coverage waypoints for the indoor Crazyflie demonstration.

The certified planner (sar/planner.py) is run offline on a small floor grid:
2.0 m x 2.0 m, 0.4 m cells, one UAV, a virtual no-fly zone (NFZ) box, and a
2 s epoch at 0.5 m/s. The result is a list of waypoints (grid frame, metres)
that never enter the NFZ inflated by a safety margin, plus the per-epoch
certificate F/UB for each decision.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from sar.objective import DetectionModel, EpochObjective, bayes_update  # noqa: E402
from sar.planner import plan_epoch  # noqa: E402
from sar.reachable import reachable_cells  # noqa: E402


@dataclass
class FloorSetup:
    size: float = 2.0            # square floor area, metres
    cell: float = 0.4            # grid cell, metres
    nfz: tuple = (1.2, 0.4, 2.0, 1.2)   # x0, y0, x1, y1 of the NFZ box, metres
    nfz_margin: float = 0.15     # extra clearance kept by the planner
    start: tuple = (0.2, 0.2)    # take-off point (a cell centre), metres
    speed: float = 0.5           # planning speed, m/s
    epoch: float = 2.0           # tactical epoch, s
    epochs: int = 14             # number of decisions
    hotspot: tuple = (0.6, 1.6)  # higher prior probability near this point
    footprint_radius: float = 0.45
    footprint_sigma: float = 0.40
    q0: float = 0.85


@dataclass
class Plan:
    waypoints: list = field(default_factory=list)   # decision waypoints [(x, y)], incl. start
    route: list = field(default_factory=list)       # flown route through grid cells [(x, y, decision_index)]
    certified: list = field(default_factory=list)   # F/UB per epoch
    expected_detections: float = 0.0
    prior_mass: float = 0.0


def box_clearance(x, y, box):
    """Signed distance from point to an axis-aligned box (negative = inside)."""
    x0, y0, x1, y1 = box
    dx = max(x0 - x, 0.0, x - x1)
    dy = max(y0 - y, 0.0, y - y1)
    if dx == 0.0 and dy == 0.0:
        return -min(x - x0, x1 - x, y - y0, y1 - y)
    return float(np.hypot(dx, dy))


def grid_path(a, b, n, blocked):
    """Shortest 8-connected path of cells from a to b; a diagonal step is
    allowed only if both cells it cuts past are free (no corner cutting)."""
    import heapq
    dist, prev, h = {a: 0.0}, {}, [(0.0, a)]
    while h:
        d, u = heapq.heappop(h)
        if u == b:
            break
        if d > dist[u]:
            continue
        r, c = divmod(u, n)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr, cc = r + dr, c + dc
                if (dr or dc) and 0 <= rr < n and 0 <= cc < n:
                    v = rr * n + cc
                    if blocked[v] or (dr and dc and (blocked[r * n + cc] or blocked[rr * n + c])):
                        continue
                    nd = d + (1.4142135623730951 if dr and dc else 1.0)
                    if nd < dist.get(v, 1e18):
                        dist[v], prev[v] = nd, u
                        heapq.heappush(h, (nd, v))
    path = [b]
    while path[-1] != a:
        path.append(prev[path[-1]])
    return path[::-1]


def make_plan(s: FloorSetup = FloorSetup()) -> Plan:
    n = int(round(s.size / s.cell))
    model = DetectionModel(n, n, s.cell, q0=s.q0, sigma=s.footprint_sigma,
                           radius=s.footprint_radius)
    cx, cy = model.centres[:, 0], model.centres[:, 1]
    blocked = np.array([box_clearance(x, y, s.nfz) < s.nfz_margin for x, y in zip(cx, cy)])
    # Prior: survivor somewhere on the free floor, more likely near the hotspot.
    w = np.exp(-((cx - s.hotspot[0]) ** 2 + (cy - s.hotspot[1]) ** 2) / (2 * 0.6 ** 2)) + 0.3
    w[blocked] = 0.0
    p = 0.9 * w / w.sum()
    plan = Plan(prior_mass=float(p.sum()))
    cur = int(np.argmin((cx - s.start[0]) ** 2 + (cy - s.start[1]) ** 2))
    plan.waypoints.append((float(cx[cur]), float(cy[cur])))
    plan.route.append((float(cx[cur]), float(cy[cur]), 0))
    for _ in range(s.epochs):
        reach = reachable_cells(cur, n, n, s.cell, blocked, s.speed * s.epoch)
        A = sorted(reach)
        obj = EpochObjective(model, p)
        # Fallback target: nearest free cell with residual mass, one hop at a time.
        resid = np.where(blocked, 0, p)
        target = int(np.argmax(resid / (1e-3 + np.hypot(cx - cx[cur], cy - cy[cur]))))
        hop = min(A, key=lambda a: np.hypot(cx[a] - cx[target], cy[a] - cy[target]))
        d = plan_epoch(obj, [A], use_hungarian=False, epsilon=0.02, fallback_targets=[hop])
        nxt = d.actions[0]
        plan.certified.append(d.certified_ratio)
        plan.expected_detections += d.value
        p = bayes_update(model, p, nxt)
        for c in grid_path(cur, nxt, n, blocked)[1:]:
            plan.route.append((float(cx[c]), float(cy[c]), len(plan.waypoints)))
        cur = nxt
        plan.waypoints.append((float(cx[cur]), float(cy[cur])))
    # Safety check of every straight segment against the inflated NFZ.
    pts = [(x, y) for x, y, _ in plan.route]
    for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
        for t in np.linspace(0, 1, 25):
            x, y = x0 + t * (x1 - x0), y0 + t * (y1 - y0)
            assert box_clearance(x, y, s.nfz) >= s.nfz_margin - 1e-9, "segment enters NFZ margin"
    return plan


if __name__ == "__main__":
    pl = make_plan()
    for i, (x, y) in enumerate(pl.waypoints):
        print(f"{i:2d}  x={x:.2f}  y={y:.2f}" + (f"  F/UB={pl.certified[i-1]:.2f}" if i else "  (take-off)"))
    print(f"route: {len(pl.route)} grid points, length {sum(np.hypot(b[0]-a[0], b[1]-a[1]) for a, b in zip(pl.route[:-1], pl.route[1:])):.2f} m")
    print(f"expected detections {pl.expected_detections:.2f} of prior mass {pl.prior_mass:.2f}; "
          f"min certified ratio {min(pl.certified):.2f}")
