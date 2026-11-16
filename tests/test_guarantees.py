"""Brute-force checks of the per-epoch guarantees on small random instances.

Run:  python -m pytest -q tests   (or python tests/test_guarantees.py)
"""
import itertools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sar.objective import DetectionModel, EpochObjective, value_of, bayes_update  # noqa: E402
from sar.planner import plan_epoch, sequential_greedy, upper_bound  # noqa: E402
from sar.reachable import NFZ, blocked_mask, reachable_cells  # noqa: E402


def random_instance(rng, n=8, m=3, reach=6.0):
    model = DetectionModel(n, n, cell_size=1.0, q0=rng.uniform(0.5, 0.95),
                           sigma=rng.uniform(0.8, 2.5), radius=rng.uniform(1.0, 2.5))
    p = rng.dirichlet(np.ones(n * n) * 0.3) * rng.uniform(1, 5)
    p = np.clip(p, 0, 0.99)
    nfzs = [NFZ(rng.uniform(0, n), rng.uniform(0, n), rng.uniform(0.5, 1.5)) for _ in range(2)]
    blocked = blocked_mask(model.centres, nfzs, margin=0.2)
    free = np.flatnonzero(~blocked)
    starts = rng.choice(free, size=m, replace=False)
    sets = [sorted(reachable_cells(int(s), n, n, 1.0, blocked, reach)) for s in starts]
    return model, p, sets


def brute_opt(model, p, sets):
    best = 0.0
    for combo in itertools.product(*sets):
        best = max(best, value_of(model, p, combo))
    return best


def test_submodular_and_monotone():
    rng = np.random.default_rng(0)
    for _ in range(200):
        model, p, _ = random_instance(rng)
        cells = rng.choice(model.n_cells, size=6, replace=False)
        A, B, x = list(cells[:2]), list(cells[:4]), int(cells[5])  # A subset of B
        gA = value_of(model, p, A + [x]) - value_of(model, p, A)
        gB = value_of(model, p, B + [x]) - value_of(model, p, B)
        assert gA >= gB - 1e-12            # diminishing returns
        assert gB >= -1e-12                # monotone


def test_not_modular():
    model = DetectionModel(5, 5, 1.0, q0=0.9, sigma=2.0, radius=2.0)
    p = np.full(25, 0.5)
    single = value_of(model, p, [12])
    assert value_of(model, p, [12, 12]) < 2 * single - 1e-9  # overlap -> diminishing


def test_greedy_half_and_bound():
    rng = np.random.default_rng(1)
    worst, worst_cert = 1.0, 1.0
    for _ in range(150):
        model, p, sets = random_instance(rng, n=7, m=3, reach=2.5)
        opt = brute_opt(model, p, sets)
        d = plan_epoch(EpochObjective(model, p), sets, use_hungarian=True)
        assert d.value >= 0.5 * opt - 1e-9, (d.value, opt)
        assert d.upper_bound >= opt - 1e-9          # certificate is a valid bound
        worst = min(worst, d.value / opt if opt > 0 else 1)
        worst_cert = min(worst_cert, d.certified_ratio)
    print(f"greedy/opt worst {worst:.3f}; certified ratio worst {worst_cert:.3f}")


def test_certified_policy_bound():
    rng = np.random.default_rng(2)
    for beta in (0.5, 0.9):
        for _ in range(100):
            model, p, sets = random_instance(rng, n=7, m=3, reach=2.5)
            opt = brute_opt(model, p, sets)
            # adversarial proposer: always proposes the worst reachable cell
            prop = lambda i, o, A: A[int(np.argmin(o.gains(A)))]
            d = plan_epoch(EpochObjective(model, p), sets, proposer=prop, beta=beta)
            assert d.value >= beta / (1 + beta) * opt - 1e-9
            # random proposer
            prop2 = lambda i, o, A: A[rng.integers(len(A))]
            d2 = plan_epoch(EpochObjective(model, p), sets, proposer=prop2, beta=beta)
            assert d2.value >= beta / (1 + beta) * opt - 1e-9


def test_bayes_update_reduces_mass():
    model = DetectionModel(6, 6, 1.0)
    p = np.full(36, 0.3)
    p2 = bayes_update(model, p, 14)
    assert p2.sum() < p.sum() and np.all(p2 <= p + 1e-12) and np.all(p2 >= 0)


def test_tightening_keeps_cell_free():
    centres = np.array([[5.0, 5.0]])
    z = NFZ(2.0, 5.0, 1.0, vx=1.0)          # moves towards the cell at 1 m/s
    horizon = 1.5
    assert not blocked_mask(centres, [z], margin=0.2, horizon=horizon)[0]
    for t in np.linspace(0, horizon, 31):    # free throughout the horizon
        assert not blocked_mask(centres, [z.at(t)], margin=0.2)[0]


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            f(); print("PASS", name)
