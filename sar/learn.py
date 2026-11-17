"""Learned waypoint proposer (Section VI): Double DQN with a dueling head,
n-step returns and prioritised replay, trained on a fast kinematic version of
the mission (no MPC; UAVs move to the chosen cell), then evaluated behind the
certification gate in the full simulator.

Decision = one UAV, one epoch. Parameter sharing across UAVs. Observation:
  local window (2w+1)^2 x 4 channels: posterior p, p after the footprints of
      the UAVs that already chose this epoch, blocked mask, reachable mask;
  global map (G x G, block-averaged p) and the UAV position on it;
  scalars: remaining flight-time fraction, epoch fraction.
Action = a cell of the (2r+1)^2 reach window, masked to A_i(k).
Reward = marginal gain Delta(a | S_{i-1}) (expected detections), gamma per epoch.

Pre-registered stopping rule (paper, Sec. VI): evaluate every `eval_every`
steps on `eval_episodes` held-out episodes; stop when the held-out mean has
improved by < `tol` (relative) over the last `window` evaluations, with
min/max step budgets. Applied per seed; the campaign stops when all seeds stop.
"""
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, asdict

import numpy as np

from .objective import DetectionModel, EpochObjective, bayes_update
from .reachable import NFZ, blocked_mask, free_distances, reachable_cells
from .sim import SimConfig


# ---------------------------------------------------------------- environment
class TrainEnv:
    def __init__(self, cfg: SimConfig, seed: int, w: int = 5, G: int = 10):
        from .sim import Mission
        self.cfg, self.w, self.G = cfg, w, G
        self.m = Mission(cfg, seed)                     # reuse prior, truth, NFZ generation
        self.rng = np.random.default_rng(seed + 99)
        self.n = self.m.n
        self.reach_r = int(math.floor(cfg.v_cruise * cfg.epoch / cfg.cell))
        self.pos = [self.m.cell_of(x[:2]) for x in self.m.x]
        self.t, self.k = 0.0, 0
        self.done = False
        self._new_epoch()

    @property
    def n_actions(self):
        return (2 * self.reach_r + 1) ** 2

    def _new_epoch(self):
        c, mm = self.cfg, self.cfg.mpc
        self.blocked = blocked_mask(self.m.model.centres, self.m.nfz_now(), mm.margin, horizon=c.epoch + mm.N * mm.dt)
        self.sets, _ = self._sets()
        self.obj = EpochObjective(self.m.model, self.m.p)
        self.i = 0
        self.live = [i for i in range(c.n_uav) if self.sets[i]]
        if not self.live:
            self.done = True

    def _sets(self):
        c = self.cfg
        base = self.m.cell_of(c.base)
        d_home = free_distances(base, self.n, self.n, c.cell, self.blocked)
        out = []
        for u in range(c.n_uav):
            reach = reachable_cells(self.pos[u], self.n, self.n, c.cell, self.blocked, c.v_cruise * c.epoch)
            rem = c.endurance - self.t - c.reserve - c.epoch
            out.append([a for a, d in reach.items() if d_home[a] / c.v_cruise <= rem])
        return out, d_home

    def _act_to_cell(self, u, a):
        R = self.reach_r; r0, c0 = divmod(self.pos[u], self.n)
        dr, dc = divmod(a, 2 * R + 1)
        rr, cc = r0 + dr - R, c0 + dc - R
        if 0 <= rr < self.n and 0 <= cc < self.n:
            return rr * self.n + cc
        return None

    def mask(self):
        u = self.live[self.i]
        A = set(self.sets[u]); m = np.zeros(self.n_actions, bool)
        for a in range(self.n_actions):
            cell = self._act_to_cell(u, a)
            m[a] = cell is not None and cell in A
        return m

    def gains(self):
        """Marginal gain of every action (nan where masked) - used by the gate."""
        u = self.live[self.i]; m = self.mask(); g = np.full(self.n_actions, np.nan)
        for a in np.flatnonzero(m):
            g[a] = self.obj.gain(self._act_to_cell(u, a))
        return g

    def obs(self):
        u = self.live[self.i]
        return build_obs(self.m.p, self.obj.miss, self.blocked, self.sets[u], self.pos, u, self.t,
                         self.i, len(self.live), self.cfg, self.n, self.w, self.G)

    @property
    def obs_dim(self):
        return 4 * (2 * self.w + 1) ** 2 + 3 * self.G ** 2 + 2

    def step(self, a):
        """Apply action a for the current UAV. Returns (reward, epoch_done)."""
        u = self.live[self.i]; cell = self._act_to_cell(u, a)
        r = self.obj.add(cell)
        self.pos[u] = cell
        self.i += 1
        if self.i < len(self.live):
            return r, False
        # end of epoch: looks (truth sampling not needed for training: expected reward), Bayes update, NFZ motion
        for v in self.live:
            self.m.p = bayes_update(self.m.model, self.m.p, self.pos[v])
        steps = int(round(self.cfg.epoch / self.cfg.mpc.dt))
        for _ in range(steps):
            self.m.move_nfz(self.cfg.mpc.dt)
        self.t += self.cfg.epoch; self.m.t = self.t; self.k += 1
        if self.t >= self.cfg.t_max or self.m.p.sum() < 1e-3:
            self.done = True
        else:
            self._new_epoch()
        return r, True


def build_obs(p, miss, blocked, set_u, pos, u, t, i, n_live, cfg, n, w, G):
    """Observation of UAV u (shared by training and by the full simulator)."""
    r0, c0 = divmod(pos[u], n)
    P = p.reshape(n, n); Pm = (p * miss).reshape(n, n)
    B = blocked.reshape(n, n).astype(float)
    Rm = np.zeros(n * n); Rm[list(set_u)] = 1; Rm = Rm.reshape(n, n)
    def win(X, fill=0.0):
        Z = np.full((2 * w + 1, 2 * w + 1), fill)
        rs, re = max(0, r0 - w), min(n, r0 + w + 1); cs, ce = max(0, c0 - w), min(n, c0 + w + 1)
        Z[rs - (r0 - w):re - (r0 - w), cs - (c0 - w):ce - (c0 - w)] = X[rs:re, cs:ce]
        return Z
    scale = 1.0 / max(1e-6, p.max())
    loc = np.stack([win(P) * scale, win(Pm) * scale, win(B, 1.0), win(Rm)])
    bs = n // G
    glob = P[:G * bs, :G * bs].reshape(G, bs, G, bs).sum((1, 3)); glob = glob / max(1e-9, glob.max())
    me = np.zeros((G, G)); me[min(G - 1, r0 // bs), min(G - 1, c0 // bs)] = 1
    others = np.zeros((G, G))
    for v in range(len(pos)):
        if v != u:
            rv, cv = divmod(pos[v], n); others[min(G - 1, rv // bs), min(G - 1, cv // bs)] += 1
    sc = np.array([max(0.0, 1 - t / cfg.endurance), i / max(1, n_live)])
    return np.concatenate([loc.ravel(), np.stack([glob, me, others]).ravel(), sc]).astype(np.float32)


# ---------------------------------------------------------------- network
def make_qnet(obs_dim, n_actions, w, G, hidden=256):
    import torch
    import torch.nn as nn

    class QNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.w, self.G = w, G
            self.loc = nn.Sequential(nn.Conv2d(4, 32, 3, padding=1), nn.ReLU(), nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
                                     nn.Flatten())
            self.glob = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.Flatten())
            d = 32 * (2 * w + 1) ** 2 + 16 * G * G + 2
            self.trunk = nn.Sequential(nn.Linear(d, hidden), nn.ReLU())
            self.V = nn.Linear(hidden, 1); self.A = nn.Linear(hidden, n_actions)

        def forward(self, x):
            k = 4 * (2 * self.w + 1) ** 2; g = 3 * self.G ** 2
            L = x[:, :k].view(-1, 4, 2 * self.w + 1, 2 * self.w + 1)
            Gm = x[:, k:k + g].view(-1, 3, self.G, self.G)
            h = self.trunk(torch.cat([self.loc(L), self.glob(Gm), x[:, k + g:]], 1))
            A = self.A(h)
            return self.V(h) + A - A.mean(1, keepdim=True)
    return QNet()


@dataclass
class TrainConfig:
    seed: int = 0
    gamma: float = 0.95
    lr: float = 2.5e-4
    batch: int = 128
    buffer: int = 200_000
    n_step: int = 3
    per_alpha: float = 0.6
    per_beta0: float = 0.4
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_steps: int = 200_000
    target_every: int = 2_000
    learn_start: int = 5_000
    train_every: int = 4
    reward_scale: float = 20.0
    eval_every: int = 20_000
    eval_episodes: int = 20
    min_steps: int = 200_000
    max_steps: int = 2_000_000
    tol: float = 0.01
    window: int = 5
    env_seed0: int = 100_000     # training episodes use seeds >= 100000; held-out eval 900-949; test >= 1000


class Replay:
    def __init__(self, cap, dim, n_act, alpha):
        self.cap, self.alpha, self.k, self.full = cap, alpha, 0, False
        self.o = np.zeros((cap, dim), np.float32); self.o2 = np.zeros((cap, dim), np.float32)
        self.a = np.zeros(cap, np.int64); self.r = np.zeros(cap, np.float32); self.g = np.zeros(cap, np.float32)
        self.m2 = np.zeros((cap, n_act), bool); self.d = np.zeros(cap, np.float32); self.pr = np.zeros(cap, np.float64)
        self.pmax = 1.0

    def add(self, o, a, r, g, o2, m2, d):
        k = self.k
        self.o[k], self.a[k], self.r[k], self.g[k], self.o2[k], self.m2[k], self.d[k] = o, a, r, g, o2, m2, d
        self.pr[k] = self.pmax
        self.k = (k + 1) % self.cap; self.full |= self.k == 0

    @property
    def size(self):
        return self.cap if self.full else self.k

    def sample(self, b, beta, rng):
        P = self.pr[:self.size] ** self.alpha; P /= P.sum()
        idx = rng.choice(self.size, b, p=P)
        wts = (self.size * P[idx]) ** (-beta); wts /= wts.max()
        return idx, wts


def evaluate(policy_fn, cfg: SimConfig, seeds, gate_beta=None):
    """Mean expected detections per episode on the kinematic env (held-out)."""
    tot = []
    for s in seeds:
        env = TrainEnv(cfg, s); R = 0.0
        while not env.done:
            m = env.mask(); a = policy_fn(env.obs(), m)
            if gate_beta is not None:
                g = env.gains(); best = np.nanmax(g)
                if not (g[a] >= gate_beta * best):
                    a = int(np.nanargmax(g))
            r, _ = env.step(a); R += r
        tot.append(R)
    return float(np.mean(tot))


def train(tc: TrainConfig, cfg: SimConfig, out: str, log_every=5000):
    import torch
    import torch.nn.functional as Fn
    torch.manual_seed(tc.seed); rng = np.random.default_rng(tc.seed)
    torch.set_num_threads(max(1, int(os.environ.get("TORCH_THREADS", "2"))))
    os.makedirs(out, exist_ok=True)
    env = TrainEnv(cfg, tc.env_seed0 + tc.seed * 100_000)
    q = make_qnet(env.obs_dim, env.n_actions, env.w, env.G); qt = make_qnet(env.obs_dim, env.n_actions, env.w, env.G)
    qt.load_state_dict(q.state_dict())
    opt = torch.optim.Adam(q.parameters(), lr=tc.lr)
    buf = Replay(tc.buffer, env.obs_dim, env.n_actions, tc.per_alpha)
    ep = 0; step = 0; curve = []; best = -np.inf; t0 = time.time()
    pend = {}   # uav -> list of (obs, action, reward) awaiting its next decision
    def greedy_q(o, m):
        with torch.no_grad():
            v = q(torch.from_numpy(o[None]))[0].numpy()
        v[~m] = -np.inf; return int(np.argmax(v))
    eval_seeds = list(range(900, 900 + tc.eval_episodes))
    while step < tc.max_steps:
        if env.done:
            for lst in pend.values():        # flush truncated n-step returns at episode end
                _flush_terminal(buf, lst, tc)
            pend = {}; ep += 1
            env = TrainEnv(cfg, tc.env_seed0 + tc.seed * 100_000 + ep)
        u = env.live[env.i]; o = env.obs(); m = env.mask()
        lst = pend.setdefault(u, [])
        if len(lst) == tc.n_step:             # oldest pending decision of this UAV now has its n-th successor
            R = sum((tc.gamma ** j) * lst[j][2] for j in range(tc.n_step))
            buf.add(lst[0][0], lst[0][1], R, tc.gamma ** tc.n_step, o, m, 0.0); lst.pop(0)
        eps = max(tc.eps_end, tc.eps_start - (tc.eps_start - tc.eps_end) * step / tc.eps_steps)
        a = int(rng.choice(np.flatnonzero(m))) if rng.random() < eps else greedy_q(o, m)
        r, _ = env.step(a)
        pend.setdefault(u, []).append((o, a, r * tc.reward_scale))
        step += 1
        if step > tc.learn_start and step % tc.train_every == 0 and buf.size > tc.batch:
            beta = min(1.0, tc.per_beta0 + (1 - tc.per_beta0) * step / tc.max_steps)
            idx, wts = buf.sample(tc.batch, beta, rng)
            O = torch.from_numpy(buf.o[idx]); O2 = torch.from_numpy(buf.o2[idx]); A = torch.from_numpy(buf.a[idx])
            Rw = torch.from_numpy(buf.r[idx]); Gm = torch.from_numpy(buf.g[idx]); D = torch.from_numpy(buf.d[idx])
            M2 = torch.from_numpy(buf.m2[idx])
            with torch.no_grad():                       # Double DQN target with masked argmax
                qn = q(O2).masked_fill(~M2, -1e9); an = qn.argmax(1, keepdim=True)
                y = Rw + Gm * (1 - D) * qt(O2).gather(1, an).squeeze(1)
            qa = q(O).gather(1, A[:, None]).squeeze(1)
            td = y - qa
            loss = (torch.from_numpy(wts.astype(np.float32)) * Fn.smooth_l1_loss(qa, y, reduction="none")).mean()
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(q.parameters(), 10.0); opt.step()
            buf.pr[idx] = np.abs(td.detach().numpy()) + 1e-4; buf.pmax = max(buf.pmax, float(buf.pr[idx].max()))
        if step % tc.target_every == 0:
            qt.load_state_dict(q.state_dict())
        if step % tc.eval_every == 0:
            v_un = evaluate(greedy_q, cfg, eval_seeds)
            curve.append(dict(step=step, eval=v_un, episodes=ep, wall_s=time.time() - t0))
            json.dump(dict(train=asdict(tc), curve=curve), open(os.path.join(out, "curve.json"), "w"), indent=1)
            if v_un > best:
                best = v_un; torch.save(q.state_dict(), os.path.join(out, "best.pt"))
            torch.save(q.state_dict(), os.path.join(out, "last.pt"))
            print(f"seed {tc.seed} step {step} eval {v_un:.3f} best {best:.3f} eps {eps:.2f} ep {ep} {time.time() - t0:.0f}s", flush=True)
            if step >= tc.min_steps and len(curve) > tc.window:
                ref = curve[-tc.window - 1]["eval"]; now = max(c["eval"] for c in curve[-tc.window:])
                if now - ref < tc.tol * abs(ref):
                    print(f"seed {tc.seed}: stopping rule met at step {step}", flush=True)
                    break
    return curve


def _flush_terminal(buf, lst, tc):
    for j in range(len(lst)):
        R = sum((tc.gamma ** (l - j)) * lst[l][2] for l in range(j, len(lst)))
        buf.add(lst[j][0], lst[j][1], R, 0.0, lst[j][0], np.ones(buf.m2.shape[1], bool), 1.0)
    lst.clear()


class LearnedProposer:
    """proposer(i, objective_so_far, A_i) for sar.planner.plan_epoch, using a trained network.
    The mission calls bind(...) once per epoch to give the context the observation needs."""

    def __init__(self, path, cfg: SimConfig, w: int = 5, G: int = 10):
        import torch
        self.torch, self.cfg, self.w, self.G = torch, cfg, w, G
        self.n = int(round(cfg.size / cfg.cell))
        self.R = int(math.floor(cfg.v_cruise * cfg.epoch / cfg.cell))
        obs_dim = 4 * (2 * w + 1) ** 2 + 3 * G * G + 2
        self.q = make_qnet(obs_dim, (2 * self.R + 1) ** 2, w, G)
        self.q.load_state_dict(torch.load(path, map_location="cpu")); self.q.eval()

    def bind(self, mission, live, sets, blocked):
        self.mi, self.live, self.sets, self.blocked = mission, live, sets, blocked
        self.pos = [mission.cell_of(x[:2]) for x in mission.x]

    def __call__(self, i, obj, A):
        u = self.live[i]; n, R = self.n, self.R
        o = build_obs(self.mi.p, obj.miss, self.blocked, self.sets[u], self.pos, u, self.mi.t, i, len(self.live),
                      self.cfg, n, self.w, self.G)
        with self.torch.no_grad():
            v = self.q(self.torch.from_numpy(o[None]))[0].numpy()
        r0, c0 = divmod(self.pos[u], n); Aset = set(A)
        best, arg = -np.inf, A[0]
        for a in range(len(v)):
            dr, dc = divmod(a, 2 * R + 1); rr, cc = r0 + dr - R, c0 + dc - R
            if 0 <= rr < n and 0 <= cc < n and rr * n + cc in Aset and v[a] > best:
                best, arg = v[a], rr * n + cc
        return arg
