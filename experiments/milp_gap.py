"""Mission-level optimality gap on small instances (Remark 1 of the paper).

Mission objective (expected survivors found by the end of the mission):
    J(X) = sum_c p_c(0) * (1 - exp(z_c)),   z_c = sum_{i,t,s} x_{i,t,s} log(1 - q(c, s)),
which equals the sum of the per-epoch gains F_k with Bayesian updates.
1 - exp(z) is concave, so tangent cuts give an over-estimator: the MILP
    max sum_c p_c w_c   s.t.  w_c <= 1 - exp(z_k) - exp(z_k)(z_c - z_k)  for breakpoints z_k
over the time-expanded move graph (one cell per UAV per epoch, moves to the
8-neighbours or stay, NFZ cells blocked) returns an UPPER bound UB >= OPT; J at its
solution is a feasible LOWER bound. The receding-horizon planner's J is compared with
UB (conservative gap) and LB.

  python experiments/milp_gap.py --instances 30 --grid 7 --uavs 2 --epochs 8 --out results/milp
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import scipy.sparse as sp
from scipy.optimize import Bounds, LinearConstraint, milp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sar.objective import DetectionModel, EpochObjective, bayes_update  # noqa: E402
from sar.planner import plan_epoch  # noqa: E402


def instance(seed, n, n_nfz=1):
    rng = np.random.default_rng(seed)
    model = DetectionModel(n, n, 1.0, q0=0.8, sigma=1.2, radius=1.5)
    cx, cy = model.centres.T
    w = 0.05 + sum(rng.uniform(.5, 1.5) * np.exp(-((cx - mx) ** 2 + (cy - my) ** 2) / (2 * s ** 2))
                   for mx, my, s in rng.uniform([0, 0, .8], [n, n, 2.0], (2, 3)))
    blocked = np.zeros(n * n, bool)
    for _ in range(n_nfz):
        r0, c0 = rng.integers(2, n - 2, 2)
        for dr in (0, 1):
            for dc in (0, 1):
                blocked[(r0 + dr) * n + c0 + dc] = True
    w[blocked] = 0
    p = np.clip(2.0 * w / w.sum(), 0, 0.6)
    return model, p, blocked


def neighbours(u, n, blocked):
    r, c = divmod(u, n)
    out = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            rr, cc = r + dr, c + dc
            if 0 <= rr < n and 0 <= cc < n and not blocked[rr * n + cc]:
                if dr and dc and (blocked[r * n + cc] or blocked[rr * n + c]):
                    continue
                out.append(rr * n + cc)
    return out


def mission_value(model, p0, looks):
    miss = np.ones(model.n_cells)
    for s in looks:
        cells, q = model.footprint(s)
        miss[cells] *= 1 - q
    return float(np.sum(p0 * (1 - miss)))


def receding(model, p0, starts, T, blocked, n, policy="best_of"):
    p = p0.copy(); cur = list(starts); looks = []
    for _ in range(T):
        A = [sorted(neighbours(u, n, blocked)) for u in cur]
        d = plan_epoch(EpochObjective(model, p), A, use_hungarian=(policy == "best_of"))
        cur = d.actions
        for s in cur:
            p = bayes_update(model, p, s); looks.append(s)
    return mission_value(model, p0, looks)


def solve_milp(model, p0, starts, T, blocked, n, n_cuts=24, time_limit=300):
    N, M = n * n, len(starts)
    free = np.flatnonzero(~blocked)
    idx = {}
    for i in range(M):
        for t in range(T):
            for s in free:
                idx[i, t, s] = len(idx)
    nx = len(idx); nz = N
    # variables: x (binary) | z_c (continuous, <= 0) | w_c (continuous, [0, 1])
    nv = nx + 2 * nz
    c_obj = np.zeros(nv); c_obj[nx + nz:] = -p0          # maximise sum p w
    rows, cols, vals, lo, up = [], [], [], [], []
    r = 0
    def add(entries, l, u):
        nonlocal r
        for j, v in entries:
            rows.append(r); cols.append(j); vals.append(v)
        lo.append(l); up.append(u); r += 1
    for i in range(M):
        for t in range(T):
            add([(idx[i, t, s], 1.0) for s in free], 1, 1)           # one cell per epoch
        # first move from the start cell
        nb0 = set(neighbours(starts[i], n, blocked))
        for s in free:
            if s not in nb0:
                add([(idx[i, 0, s], 1.0)], 0, 0)
        for t in range(1, T):
            for s in free:      # x_{t,s} <= sum_{u in nb(s)} x_{t-1,u}
                add([(idx[i, t, s], 1.0)] + [(idx[i, t - 1, u], -1.0) for u in neighbours(s, n, blocked)], -np.inf, 0)
    # z_c = sum x log(1-q)
    for c in range(N):
        e = [(nx + c, 1.0)]
        for i in range(M):
            for t in range(T):
                for s in free:
                    cells, q = model.footprint(s)
                    k = np.where(cells == c)[0]
                    if len(k):
                        e.append((idx[i, t, s], -np.log(1 - q[k[0]])))
        add(e, 0, 0)
    zmin = M * T * np.log(1 - model.q0) * 1.0
    for zk in np.linspace(zmin, 0, n_cuts):
        a = np.exp(zk)
        for c in range(N):     # w_c + a z_c <= 1 - a + a zk
            add([(nx + nz + c, 1.0), (nx + c, a)], -np.inf, 1 - a + a * zk)
    A = sp.csr_matrix((vals, (rows, cols)), shape=(r, nv))
    integrality = np.r_[np.ones(nx), np.zeros(2 * nz)]
    bounds = Bounds(np.r_[np.zeros(nx), np.full(nz, zmin), np.zeros(nz)], np.r_[np.ones(nx), np.zeros(nz), np.ones(nz)])
    t0 = time.time()
    res = milp(c_obj, constraints=LinearConstraint(A, lo, up), integrality=integrality, bounds=bounds,
               options=dict(time_limit=time_limit, mip_rel_gap=1e-4))
    ub = -res.fun if res.x is not None else np.nan
    if hasattr(res, "mip_dual_bound") and res.mip_dual_bound is not None and not res.success:
        ub = -res.mip_dual_bound              # not proven optimal: use the dual bound
    looks = [s for (i, t, s), j in idx.items() if res.x is not None and res.x[j] > 0.5]
    lb = mission_value(model, p0, looks) if looks else np.nan
    return dict(ub=float(ub), lb=float(lb), status=int(res.status), milp_s=time.time() - t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", type=int, default=30)
    ap.add_argument("--grid", type=int, default=7)
    ap.add_argument("--uavs", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--seed0", type=int, default=5000)
    ap.add_argument("--time-limit", type=float, default=300)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, f"milp_g{a.grid}_m{a.uavs}_t{a.epochs}.jsonl")
    with open(path, "a") as f:
        for k in range(a.instances):
            seed = a.seed0 + k
            model, p0, blocked = instance(seed, a.grid)
            starts = [0, a.grid - 1][:a.uavs] + [a.grid * (a.grid - 1)] * max(0, a.uavs - 2)
            starts = [s for s in starts]
            for s in starts:
                blocked[s] = False
            rec = dict(seed=seed, grid=a.grid, uavs=a.uavs, epochs=a.epochs)
            for pol in ("best_of", "greedy"):
                rec[pol] = receding(model, p0, starts, a.epochs, blocked, a.grid, pol)
            rec.update(solve_milp(model, p0, starts, a.epochs, blocked, a.grid, time_limit=a.time_limit))
            rec["ratio_vs_ub"] = rec["best_of"] / rec["ub"]
            rec["ratio_vs_lb"] = rec["best_of"] / rec["lb"]
            f.write(json.dumps(rec) + "\n"); f.flush()
            print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in rec.items()}), flush=True)


if __name__ == "__main__":
    main()
