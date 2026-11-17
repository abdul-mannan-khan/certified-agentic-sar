"""Multi-UAV search-and-rescue mission simulator (revision experiments).

What it models (each item answers a reviewer point):
* imperfect, range-dependent detection and an exact Bayesian posterior over
  survivor locations (R2.5); ground truth is drawn from the prior, so the
  belief is calibrated by construction, and a mismatch option tests it;
* no-fly zones that move continuously with speed <= vbar (A1), plus a
  separate pop-up stress mode that violates A1 on purpose (R2.3, R1.2);
* battery endurance with a return-to-base constraint encoded in the
  reachable sets (still a partition matroid);
* command loss on the radio link;
* a large area (default 2 km x 2 km) and several UAVs;
* a two-level loop: the certified tactical planner every epoch (sar/planner.py)
  and the safe MPC at 5 Hz (sar/nmpc.py), with an imperfect plant
  (acceleration lag and bounded gusts).

Policies: "greedy", "hungarian" (v1-style assignment alone), "best_of"
(greedy + Hungarian, keep the better), "certified" (learned proposer behind the
beta gate; any callable), "lawnmower" and "random".
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from typing import Callable, Optional

import numpy as np

from .nmpc import MPCParams, SafeMPC
from .objective import DetectionModel, EpochObjective, bayes_update
from .planner import plan_epoch, sequential_greedy, upper_bound
from .reachable import NFZ, blocked_mask, free_distances, grid_path, reachable_cells
from .plants import ModelPlant


@dataclass
class SimConfig:
    size: float = 2000.0          # m, square area
    cell: float = 50.0            # m
    n_uav: int = 4
    base: tuple = (25.0, 25.0)    # m, launch/recovery point
    v_cruise: float = 10.0        # m/s, planning speed
    epoch: float = 10.0           # s, tactical epoch Delta
    t_max: float = 1800.0         # s, mission limit
    endurance: float = 1500.0     # s, battery
    reserve: float = 60.0         # s, kept for landing
    q0: float = 0.9
    sigma: float = 70.0           # m
    radius: float = 100.0         # m, footprint
    q_mismatch: float = 1.0       # true q = q_mismatch * modelled q (1 = matched)
    n_survivor_expected: float = 8.0
    n_clusters: int = 3
    n_nfz: int = 4
    nfz_r: tuple = (100.0, 180.0)
    vbar: float = 3.0             # m/s, NFZ speed bound (A1)
    popup: bool = False           # stress mode: NFZs appear without warning
    p_cmd_loss: float = 0.05
    acc_lag: float = 0.15         # s, plant acceleration time constant
    gust: float = 0.3             # m/s^2, bounded disturbance
    beta: float = 0.9
    epsilon: float = 0.001        # tuned on seeds 1-3 only (fallback 2 %); evaluation seeds >= 1000
    mpc: MPCParams = field(default_factory=MPCParams)


@dataclass
class EpisodeResult:
    policy: str
    seed: int
    n_survivors: int
    found_times: list
    found: int
    t_end: float
    incursion: bool
    min_clearance: float
    certified_ratio: list
    accept_rate: float
    fallback_rate: float
    mpc_modes: dict
    plan_ms: list
    mpc_ms_p95: float
    rtb: int
    prior_mass: float
    detected_mass: float          # sum over epochs of F_k(S_k) (expected detections)


class Mission:
    def __init__(self, cfg: SimConfig, seed: int, plant=None, record: bool = False):
        self.cfg, self.rng = cfg, np.random.default_rng(seed)
        self.record, self.trace, self.nfz_trace = record, [], []
        self.seed = seed
        c = cfg
        self.n = int(round(c.size / c.cell))
        self.model = DetectionModel(self.n, self.n, c.cell, q0=c.q0, sigma=c.sigma, radius=c.radius)
        self.true_model = DetectionModel(self.n, self.n, c.cell, q0=min(1.0, c.q0 * c.q_mismatch),
                                         sigma=c.sigma, radius=c.radius)
        cx, cy = self.model.centres.T
        # Prior: mixture of clusters (last-known positions), Bernoulli occupancy per cell.
        w = np.full(self.n * self.n, 0.05)
        for _ in range(c.n_clusters):
            mx, my = self.rng.uniform(0.15, 0.95, 2) * c.size
            s = self.rng.uniform(0.06, 0.15) * c.size
            w += self.rng.uniform(0.5, 1.5) * np.exp(-((cx - mx) ** 2 + (cy - my) ** 2) / (2 * s * s))
        p = c.n_survivor_expected * w / w.sum()
        self.p = np.clip(p, 0, 0.6)
        self.truth = self.rng.random(self.p.size) < self.p      # ground truth drawn from the prior
        self.found = np.zeros_like(self.truth)
        # NFZs
        self.nfz = []
        for _ in range(c.n_nfz):
            r = self.rng.uniform(*c.nfz_r)
            while True:
                x, y = self.rng.uniform(0.2, 0.95, 2) * c.size
                if np.hypot(x - c.base[0], y - c.base[1]) > r + 400:
                    break
            hd = self.rng.uniform(0, 2 * np.pi); sp_ = self.rng.uniform(0.5, 1.0) * c.vbar
            self.nfz.append(NFZ(x, y, r, sp_ * np.cos(hd), sp_ * np.sin(hd)))
        self.popup_t = sorted(self.rng.uniform(200, 900, c.n_nfz)) if c.popup else None
        self.active = [not c.popup] * c.n_nfz
        # UAVs: state [px, py, vx, vy], plant acceleration state
        x0 = np.array([[c.base[0] + 15 * i, c.base[1], 0, 0] for i in range(c.n_uav)], float)
        # plant: internal model by default; PX4 SITL / HIL / AirSim plants live in sar/plants.py
        self.plant = plant if plant is not None else ModelPlant(x0, c.mpc.dt, c.acc_lag, c.gust, np.random.default_rng(seed + 7))
        self.x = self.plant.states()
        self.mpc = [SafeMPC(c.mpc) for _ in range(c.n_uav)]
        self.home = [False] * c.n_uav
        self.t = 0.0

    # ----- environment pieces -----
    def cell_of(self, xy):
        c = self.cfg
        col = int(np.clip(xy[0] // c.cell, 0, self.n - 1)); row = int(np.clip(xy[1] // c.cell, 0, self.n - 1))
        return row * self.n + col

    def nfz_now(self):
        return [z for z, a in zip(self.nfz, self.active) if a]

    def move_nfz(self, dt):
        c = self.cfg
        for k, z in enumerate(self.nfz):
            if self.popup_t is not None and not self.active[k] and self.t >= self.popup_t[k]:
                self.active[k] = True
            # slow heading random walk, speed kept <= vbar
            hd = np.arctan2(z.vy, z.vx) + self.rng.normal(0, 0.02)
            sp_ = min(c.vbar, z.speed)
            z.vx, z.vy = sp_ * np.cos(hd), sp_ * np.sin(hd)
            z.x += z.vx * dt; z.y += z.vy * dt
            if not (0 < z.x < c.size):
                z.vx *= -1
            if not (0 < z.y < c.size):
                z.vy *= -1

    def sense(self, i):
        """Look from UAV i's current cell; returns detected cells."""
        s = self.cell_of(self.x[i, :2])
        cells, q = self.true_model.footprint(s)
        hit = cells[(self.truth[cells] & ~self.found[cells]) & (self.rng.random(len(cells)) < q)]
        self.found[hit] = True
        self.p = bayes_update(self.model, self.p, s, detected_cells=hit)
        return hit

    # ----- tactical level -----
    def action_sets(self, blocked):
        c = self.cfg
        base_cell = self.cell_of(c.base)
        d_home = free_distances(base_cell, self.n, self.n, c.cell, blocked)
        sets = []
        for i in range(c.n_uav):
            cur = self.cell_of(self.x[i, :2])
            if self.home[i]:
                sets.append(None); continue
            reach = reachable_cells(cur, self.n, self.n, c.cell, blocked, c.v_cruise * c.epoch)
            remaining = c.endurance - self.t - c.reserve - c.epoch
            A = [a for a, d in reach.items() if d_home[a] / c.v_cruise <= remaining]
            sets.append(A if A else None)
        return sets, d_home

    def run(self, policy: str = "best_of", proposer: Optional[Callable] = None) -> EpisodeResult:
        c, m = self.cfg, self.cfg.mpc
        steps_per_epoch = int(round(c.epoch / m.dt))
        cx, cy = self.model.centres.T
        refs = [None] * c.n_uav
        found_times, ratios, plan_ms, mpc_ms = [], [], [], []
        acc_n = acc_k = fb_n = 0
        min_clear, incursion, rtb, det_mass = np.inf, False, 0, 0.0
        prior_mass = float(self.p.sum())
        lawn = self._lawnmower_routes() if policy == "lawnmower" else None
        while self.t < c.t_max and not all(self.home):
            blocked = blocked_mask(self.model.centres, self.nfz_now(), m.margin, horizon=c.epoch + m.N * m.dt)
            sets, d_home = self.action_sets(blocked)
            live = [i for i in range(c.n_uav) if sets[i] is not None]
            for i in range(c.n_uav):
                if sets[i] is None and not self.home[i]:
                    refs[i] = self._path_ref(i, self.cell_of(c.base), blocked, land=True); rtb += 1
                    self.home[i] = True
            if live:
                obj = EpochObjective(self.model, self.p)
                A = [sets[i] for i in live]
                t0 = time.perf_counter()
                if policy in ("greedy", "best_of", "certified"):
                    resid = np.where(blocked, 0.0, self.p)
                    fb = []
                    for i in live:
                        cur = self.cell_of(self.x[i, :2])
                        tgt = int(np.argmax(resid / (c.cell + np.hypot(cx - cx[cur], cy - cy[cur]))))
                        fb.append(min(sets[i], key=lambda a: np.hypot(cx[a] - cx[tgt], cy[a] - cy[tgt])))
                    d = plan_epoch(obj, A, proposer=proposer if policy == "certified" else None,
                                   beta=c.beta, use_hungarian=(policy == "best_of"),
                                   epsilon=c.epsilon, fallback_targets=fb)
                    acts = d.actions; ratios.append(d.certified_ratio)
                    acc_n += sum(d.accepted); acc_k += len(d.accepted); fb_n += sum(d.fallback)
                    det_mass += d.value
                elif policy == "hungarian":
                    from .planner import hungarian
                    acts, o = hungarian(obj, A)
                    if acts is None:
                        acts, o = sequential_greedy(obj, A)
                    ub = upper_bound(o, A); ratios.append(o.value() / ub if ub > 0 else 1.0); det_mass += o.value()
                elif policy == "random":
                    acts = [int(self.rng.choice(a)) for a in A]
                    o = obj.copy(); [o.add(a) for a in acts]
                    ub = upper_bound(o, A); ratios.append(o.value() / ub if ub > 0 else 1.0); det_mass += o.value()
                elif policy == "lawnmower":
                    acts = [self._lawn_next(lawn, i, sets[i]) for i in live]
                    o = obj.copy(); [o.add(a) for a in acts]
                    ub = upper_bound(o, A); ratios.append(o.value() / ub if ub > 0 else 1.0); det_mass += o.value()
                else:
                    raise ValueError(policy)
                plan_ms.append(1e3 * (time.perf_counter() - t0))
                for i, a in zip(live, acts):
                    if refs[i] is None or self.rng.random() >= c.p_cmd_loss:   # command delivered
                        refs[i] = self._path_ref(i, a, blocked)
            # ----- MPC loop for one epoch -----
            for _ in range(steps_per_epoch):
                zs = [(z.x, z.y, z.r) for z in self.nfz_now()]
                for i in range(c.n_uav):
                    if refs[i] is None:
                        continue
                    ref = refs[i][:m.N]
                    if len(ref) < m.N:
                        ref = np.vstack([ref, np.tile(ref[-1], (m.N - len(ref), 1))]) if len(ref) else np.tile(np.r_[self.x[i, :2], 0, 0], (m.N, 1))
                    t0 = time.perf_counter()
                    a, info = self.mpc[i].step(self.x[i], ref, zs)
                    mpc_ms.append(time.perf_counter() - t0)
                    refs[i] = refs[i][1:] if len(refs[i]) > 1 else refs[i]
                    self.plant.apply(i, a, self.x[i])
                    if self.record:
                        self.trace.append((self.t, i, *self.x[i], *a, *ref[0], info["mode"]))
                self.plant.advance(m.dt)          # real plants block until simulated/real time has advanced
                self.x = self.plant.states()
                self.move_nfz(m.dt)
                self.t += m.dt
                if self.record:
                    self.nfz_trace.append((self.t, [(z.x, z.y, z.r) for z in self.nfz_now()]))
                for i in range(c.n_uav):
                    for z in self.nfz_now():
                        dcl = np.hypot(self.x[i, 0] - z.x, self.x[i, 1] - z.y) - z.r
                        min_clear = min(min_clear, dcl)
                        incursion |= dcl < 0
            for i in range(c.n_uav):
                if not self.home[i]:
                    for _ in self.sense(i):
                        found_times.append(self.t)
            if self.found.sum() == self.truth.sum() and self.truth.sum() > 0:
                break
        modes = {k: sum(mp.stats[k] for mp in self.mpc) for k in ("solved", "stored", "backup")}
        return EpisodeResult(policy=policy, seed=self.seed, n_survivors=int(self.truth.sum()),
                             found_times=found_times, found=int(self.found.sum()), t_end=self.t,
                             incursion=bool(incursion), min_clearance=float(min_clear),
                             certified_ratio=ratios, accept_rate=acc_n / acc_k if acc_k else float("nan"),
                             fallback_rate=fb_n / acc_k if acc_k else float("nan"), mpc_modes=modes,
                             plan_ms=plan_ms, mpc_ms_p95=float(1e3 * np.percentile(mpc_ms, 95)) if mpc_ms else 0.0,
                             rtb=rtb, prior_mass=prior_mass, detected_mass=det_mass)

    # ----- helpers -----
    def _path_ref(self, i, target, blocked, land=False):
        """Time-parametrised reference along the free grid path at cruise speed."""
        c, m = self.cfg, self.cfg.mpc
        cur = self.cell_of(self.x[i, :2])
        path = grid_path(cur, target, self.n, self.n, blocked)
        pts = [self.x[i, :2].copy()] + [self.model.centres[k] for k in path[1:]]
        if len(pts) == 1:
            pts.append(self.model.centres[target])
        pts = np.array(pts)
        seg = np.diff(pts, axis=0); L = np.linalg.norm(seg, axis=1); s_tot = L.sum()
        n_steps = max(1, int(np.ceil(s_tot / (c.v_cruise * m.dt))))
        s = np.minimum(np.arange(1, n_steps + 1) * c.v_cruise * m.dt, s_tot)
        cum = np.r_[0, np.cumsum(L)]
        out = []
        for sk in s:
            j = min(int(np.searchsorted(cum, sk, side="right")) - 1, len(L) - 1)
            u = seg[j] / max(L[j], 1e-9)
            pos = pts[j] + u * (sk - cum[j])
            vel = u * c.v_cruise if sk < s_tot - 1e-6 else np.zeros(2)
            out.append(np.r_[pos, vel])
        return np.array(out)

    def _lawnmower_routes(self):
        """Boustrophedon strips split across UAVs (classical baseline)."""
        n, routes = self.n, []
        rows_per = int(np.ceil(n / self.cfg.n_uav))
        for i in range(self.cfg.n_uav):
            seq = []
            for r in range(i * rows_per, min(n, (i + 1) * rows_per), 2):   # every second row (footprint ~2 cells)
                cols = range(n) if ((r // 2) % 2 == 0) else range(n - 1, -1, -1)
                seq += [r * n + col for col in cols]
            routes.append(seq)
        return {"routes": routes, "idx": [0] * self.cfg.n_uav}

    def _lawn_next(self, lawn, i, A):
        cx, cy = self.model.centres.T
        route, Aset = lawn["routes"][i], set(A)
        k = lawn["idx"][i]
        while k < len(route) - 1 and route[k + 1] in Aset:   # go as far along the strip as one epoch allows
            k += 1
        lawn["idx"][i] = k
        tgt = route[k]
        return min(A, key=lambda a: np.hypot(cx[a] - cx[tgt], cy[a] - cy[tgt]))


def run_episode(policy: str, seed: int, cfg: Optional[SimConfig] = None, proposer=None) -> dict:
    r = Mission(cfg or SimConfig(), seed).run(policy, proposer)
    return asdict(r)
