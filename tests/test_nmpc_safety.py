"""Safety tests for the MPC layer (Theorem 5): no NFZ incursion for any NFZ
motion with speed <= vbar, including adversarial pursuit of the UAV."""
import numpy as np
import pytest

from sar.nmpc import MPCParams, SafeMPC


def _run(nfz_policy, steps=900, seed=0):
    rng = np.random.default_rng(seed)
    m = SafeMPC(MPCParams()); p = m.p
    x = np.zeros(4); goal = np.array([800.0, 0.0])
    z = np.array([400.0, rng.uniform(-60, 60)]); r = 120.0
    worst = np.inf
    for _ in range(steps):
        a, _ = m.step(x, np.tile(np.r_[goal, 0, 0], (p.N, 1)), [(z[0], z[1], r)])
        x = m.A @ x + m.B @ a
        z = z + nfz_policy(x, z, rng) * p.dt
        worst = min(worst, np.hypot(x[0] - z[0], x[1] - z[1]) - r)
    return worst, m.stats


def head_on(x, z, rng): return np.array([-3.0, 0.0])
def pursuit(x, z, rng):
    d = x[:2] - z; return 3.0 * d / max(np.linalg.norm(d), 1e-9)
def random_walk(x, z, rng):
    v = rng.normal(0, 1, 2); return 3.0 * v / max(np.linalg.norm(v), 1e-9)


@pytest.mark.parametrize("pol", [head_on, pursuit, random_walk])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_no_incursion(pol, seed):
    worst, stats = _run(pol, seed=seed)
    assert worst >= 0.0, (worst, stats)


def test_terminal_cost_is_dare():
    m = SafeMPC()
    A, B, Q, R, P = m.A, m.B, m.Q, m.R, m.P
    K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
    assert np.allclose(P, Q + A.T @ P @ A - A.T @ P @ B @ K, atol=1e-8)
