"""Probabilistic detection objective for multi-UAV search.

The search area is a grid of N cells. p[c] is the current probability that
cell c holds a survivor who has not yet been detected. A UAV observing from
waypoint cell s detects a survivor in cell c with probability q(c, s), which
decreases with the distance between the two cell centres.

For a set S of sensing actions taken in one epoch, the expected number of new
detections is

    F(S) = sum_c p[c] * (1 - prod_{s in S} (1 - q(c, s))),

which is monotone and submodular in S (Lemma 1 in the paper).
"""
from __future__ import annotations

import numpy as np


class DetectionModel:
    """Range-dependent detection probability on a square grid.

    q(c, s) = q0 * exp(-(d / sigma)^2) for d <= radius, and 0 beyond, where d
    is the distance in metres between the centres of cells c and s.
    """

    def __init__(self, n_rows: int, n_cols: int, cell_size: float,
                 q0: float = 0.9, sigma: float = 2.0, radius: float = 3.0):
        if not 0.0 < q0 <= 1.0:
            raise ValueError("q0 must lie in (0, 1]")
        self.n_rows, self.n_cols, self.cell_size = n_rows, n_cols, cell_size
        self.q0, self.sigma, self.radius = q0, sigma, radius
        self.n_cells = n_rows * n_cols
        rows, cols = np.divmod(np.arange(self.n_cells), n_cols)
        self.centres = np.stack([(cols + 0.5) * cell_size,
                                 (rows + 0.5) * cell_size], axis=1)
        # Footprint offsets: every cell within `radius` of a waypoint cell.
        r_cells = int(np.ceil(radius / cell_size))
        offs, probs = [], []
        for dr in range(-r_cells, r_cells + 1):
            for dc in range(-r_cells, r_cells + 1):
                d = cell_size * np.hypot(dr, dc)
                if d <= radius + 1e-9:
                    offs.append((dr, dc))
                    probs.append(q0 * np.exp(-(d / sigma) ** 2))
        self._offs = np.array(offs, dtype=int)
        self._probs = np.array(probs)
        self.q_min = float(self._probs.min())

    def footprint(self, s: int) -> tuple[np.ndarray, np.ndarray]:
        """Cells observed from waypoint cell s and their detection probabilities."""
        r, c = divmod(int(s), self.n_cols)
        rr, cc = r + self._offs[:, 0], c + self._offs[:, 1]
        ok = (rr >= 0) & (rr < self.n_rows) & (cc >= 0) & (cc < self.n_cols)
        return rr[ok] * self.n_cols + cc[ok], self._probs[ok]


class EpochObjective:
    """F_k(S) for one decision epoch, with incremental marginal gains.

    `miss[c]` holds prod_{s in S} (1 - q(c, s)) for the actions added so far,
    so the marginal gain of a new action s is sum_c p[c] * miss[c] * q(c, s).
    """

    def __init__(self, model: DetectionModel, p: np.ndarray):
        self.model = model
        self.p = np.asarray(p, dtype=float)
        self.miss = np.ones(model.n_cells)
        self.chosen: list[int] = []

    def gain(self, s: int) -> float:
        cells, q = self.model.footprint(s)
        return float(np.sum(self.p[cells] * self.miss[cells] * q))

    def gains(self, candidates) -> np.ndarray:
        return np.array([self.gain(s) for s in candidates])

    def add(self, s: int) -> float:
        g = self.gain(s)
        cells, q = self.model.footprint(s)
        self.miss[cells] *= (1.0 - q)
        self.chosen.append(int(s))
        return g

    def value(self) -> float:
        return float(np.sum(self.p * (1.0 - self.miss)))

    def copy(self) -> "EpochObjective":
        o = EpochObjective(self.model, self.p)
        o.miss = self.miss.copy()
        o.chosen = list(self.chosen)
        return o


def value_of(model: DetectionModel, p: np.ndarray, actions) -> float:
    """F(S) evaluated from scratch (used by tests and exact baselines)."""
    o = EpochObjective(model, p)
    for s in actions:
        o.add(s)
    return o.value()


def bayes_update(model: DetectionModel, p: np.ndarray, s: int,
                 detected_cells=()) -> np.ndarray:
    """Posterior after looking from s without detecting anyone in the footprint.

    For cells with a confirmed detection the survivor is found, so their
    residual undetected probability is set to zero.
    """
    p = np.asarray(p, dtype=float).copy()
    cells, q = model.footprint(s)
    p[cells] = p[cells] * (1.0 - q) / (1.0 - p[cells] * q)
    for c in detected_cells:
        p[c] = 0.0
    return p
