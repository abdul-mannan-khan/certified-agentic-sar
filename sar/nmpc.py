"""Safe tracking MPC for one UAV among moving no-fly zones (Section V).

Model. The outer loop plans horizontal translation; the autopilot (PX4)
holds altitude and attitude. Per UAV the prediction model is the discrete
double integrator

    p+ = p + v dt + a dt^2 / 2,     v+ = v + a dt,

with |a| <= a_max and |v| <= v_max (inscribed octagons, so the true norm
bounds hold).

No-fly zones. NFZ j is a disc of radius r_j whose centre moves at speed at
most vbar_j (Assumption A1); only its current position is used (A2). At
prediction step l the UAV must stay outside the disc of radius

    r_j + m + vbar_j * l * dt                                    (tightening)

around the measured centre. Whatever the NFZ does in that time, the true disc
lies inside this one, so the plan is safe for any admissible NFZ motion.

Convexification. Each disc constraint is replaced by the half-plane tangent
to the tightened disc at the point closest to the previous plan,
n^T (p_l - c_j) >= r_j + m + vbar_j l dt. The half-plane lies outside the
disc, so the QP feasible set is inside the true one (safe), and the shifted
previous plan satisfies it by construction.

Terminal ingredients. Terminal cost P from the discrete algebraic Riccati
equation of the model with weights (Q, R); terminal constraint: the UAV is
stopped (v_N = 0) with an extra clearance d_b = vbar * t_acc + v_e^2/(2 a_max),
enough for the backup law to reach the evasion speed v_e > vbar.

Backup law kappa_b (brake-and-evade): accelerate at a_max away from the
nearest NFZ until the radial speed is v_e, then hold. Under the single-NFZ
interaction assumption (A5) the distance never drops below the margin.

Safety logic in closed loop (Theorem 5): the controller always stores a plan
whose finite part is tightened-safe and whose tail is kappa_b. A new QP
solution replaces the stored plan only if solved; otherwise the stored plan
keeps running, and when it is used up, kappa_b runs. Every applied input is
therefore a step of a plan that is safe for all admissible NFZ motions.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import osqp
import scipy.linalg as sla
import scipy.sparse as sp


@dataclass
class MPCParams:
    N: int = 25                 # horizon steps
    dt: float = 0.2             # s
    v_max: float = 12.0         # m/s
    a_max: float = 3.0          # m/s^2
    margin: float = 20.0        # m, extra clearance m
    vbar: float = 3.0           # m/s, max NFZ speed (A1)
    v_evade: float = 6.0        # m/s, backup evasion speed (> vbar)
    q_pos: float = 1.0
    q_vel: float = 0.3
    r_acc: float = 0.5
    max_nfz: int = 6            # NFZs considered per solve (nearest first)

    @property
    def backup_clearance(self) -> float:
        t_acc = self.v_evade / self.a_max
        return self.vbar * t_acc + self.v_evade ** 2 / (2 * self.a_max)


def _octagon(r: float):
    ang = np.arange(8) * np.pi / 4
    return np.stack([np.cos(ang), np.sin(ang)], 1), r  # d_k^T x <= r keeps |x| <= r / cos(pi/8)


class SafeMPC:
    """Convexified tracking MPC solved with OSQP, plus backup law."""

    def __init__(self, prm: MPCParams = MPCParams()):
        self.p = prm
        dt, N = prm.dt, prm.N
        A1 = np.array([[1, dt], [0, 1]]); B1 = np.array([[dt * dt / 2], [dt]])
        self.A = np.kron(A1, np.eye(2))          # state [px, py, vx, vy]
        self.B = np.kron(B1, np.eye(2))
        Q = np.diag([prm.q_pos] * 2 + [prm.q_vel] * 2)
        R = prm.r_acc * np.eye(2)
        self.P = sla.solve_discrete_are(self.A, self.B, Q, R)   # terminal cost
        self.Q, self.R = Q, R
        self.nx, self.nu = 4, 2
        self._build()
        self.plan_u = np.zeros((0, 2))           # stored safe plan (inputs)
        self.stats = dict(solved=0, stored=0, backup=0)

    # --- QP structure (fixed sparsity; only values change between solves) ---
    def _build(self):
        p, N, nx, nu = self.p, self.p.N, self.nx, self.nu
        nX, nU = N * nx, N * nu
        self.nz = nX + nU
        iX = lambda l: l * nx                     # x_{l+1} block, l = 0..N-1
        iU = lambda l: nX + l * nu
        Hx = sp.block_diag([sp.csc_matrix(self.Q)] * (N - 1) + [sp.csc_matrix(self.P)])
        Hu = sp.kron(sp.eye(N), sp.csc_matrix(self.R))
        self.H = sp.block_diag([Hx, Hu], format="csc")
        rows = []
        # dynamics: x_{l+1} - A x_l - B u_l = 0  (x_0 is data)
        Aeq = sp.lil_matrix((nX, self.nz))
        for l in range(N):
            Aeq[iX(l):iX(l) + nx, iX(l):iX(l) + nx] = np.eye(nx)
            if l > 0:
                Aeq[iX(l):iX(l) + nx, iX(l - 1):iX(l - 1) + nx] = -self.A
            Aeq[iX(l):iX(l) + nx, iU(l):iU(l) + nu] = -self.B
        rows.append(Aeq)
        D, _ = _octagon(1.0)
        c8 = np.cos(np.pi / 8)
        Av = sp.lil_matrix((8 * N, self.nz)); Aa = sp.lil_matrix((8 * N, self.nz))
        for l in range(N):
            Av[8 * l:8 * l + 8, iX(l) + 2:iX(l) + 4] = D
            Aa[8 * l:8 * l + 8, iU(l):iU(l) + 2] = D
        rows += [Av, Aa]
        Aterm = sp.lil_matrix((2, self.nz)); Aterm[:, iX(N - 1) + 2:iX(N - 1) + 4] = np.eye(2)
        rows.append(Aterm)
        J = p.max_nfz
        An = sp.lil_matrix((J * N, self.nz))
        for j in range(J):
            for l in range(N):
                An[j * N + l, iX(l):iX(l) + 2] = [1.0, 1.0]   # placeholders, values set per solve
        rows.append(An)
        self.Acon = sp.vstack(rows, format="csc")
        self.n_eq, self.n_v, self.n_a, self.n_t, self.n_n = nX, 8 * N, 8 * N, 2, J * N
        self.off_n = nX + 16 * N + 2
        self.l = np.concatenate([np.zeros(nX), -np.inf * np.ones(16 * N), np.zeros(2), -np.inf * np.ones(J * N)])
        self.u = np.concatenate([np.zeros(nX), p.v_max * c8 * np.ones(8 * N), p.a_max * c8 * np.ones(8 * N),
                                 np.zeros(2), np.inf * np.ones(J * N)])
        # map NFZ placeholder entries to positions in Acon.data for fast updates
        A = self.Acon.tocsc()
        self._nfz_idx = np.zeros((J, N, 2), dtype=int)
        for j in range(J):
            for l in range(N):
                r = self.off_n + j * N + l
                for k in range(2):
                    col = iX(l) + k
                    s, e = A.indptr[col], A.indptr[col + 1]
                    self._nfz_idx[j, l, k] = s + int(np.where(A.indices[s:e] == r)[0][0])
        self.Acon = A
        self.solver = osqp.OSQP()
        self.solver.setup(P=self.H, q=np.zeros(self.nz), A=self.Acon, l=self.l, u=self.u,
                          verbose=False, eps_abs=1e-5, eps_rel=1e-5, max_iter=4000, polish=True, warm_start=True)
        self._iX, self._iU = iX, iU

    def _predict(self, x0, U):
        X = [x0]
        for a in U:
            X.append(self.A @ X[-1] + self.B @ a)
        return np.array(X)

    # --- backup law ---
    def backup(self, x, nfzs):
        p = self.p
        if not nfzs:
            return np.clip(-x[2:] / p.dt, -p.a_max, p.a_max) * 0.7     # brake
        c = min(nfzs, key=lambda z: np.hypot(x[0] - z[0], x[1] - z[1]) - z[2])  # nearest zone or obstacle
        n = x[:2] - np.array(c[:2]); n /= max(np.linalg.norm(n), 1e-9)
        t = np.array([-n[1], n[0]])
        vr, vt = x[2:] @ n, x[2:] @ t
        a = (p.a_max if vr < p.v_evade else 0.0) * n - np.clip(vt / p.dt, -p.a_max, p.a_max) * 0.5 * t
        nrm = np.linalg.norm(a)
        return a if nrm <= p.a_max else a * p.a_max / nrm

    # --- one control step ---
    def step(self, x0, ref, nfzs):
        """x0: [px,py,vx,vy]; ref: (N, 4) reference states for steps 1..N;
        nfzs: list of (cx, cy, r[, vbar_j]) measured now; vbar_j = 0 for static obstacles. Returns (a, info)."""
        p, N = self.p, self.p.N
        x0 = np.asarray(x0, float)
        # nominal trajectory for linearisation: stored plan shifted (+ zeros), else hold
        U_nom = np.vstack([self.plan_u[1:], np.zeros((N, 2))])[:N] if len(self.plan_u) > 1 else np.zeros((N, 2))
        Xn = self._predict(x0, U_nom)[1:]
        near = sorted(nfzs, key=lambda z: np.hypot(x0[0] - z[0], x0[1] - z[1]) - z[2])[:p.max_nfz]
        data = self.Acon.data
        lo, up = self.l.copy(), self.u.copy()
        for j in range(p.max_nfz):
            for l in range(N):
                r = self.off_n + j * N + l
                if j < len(near):
                    cx, cy, rad = near[j][:3]
                    vb = near[j][3] if len(near[j]) > 3 else p.vbar      # per-zone speed bound (0 = static)
                    n = Xn[l, :2] - (cx, cy); nn = np.linalg.norm(n)
                    n = n / nn if nn > 1e-6 else np.array([1.0, 0.0])
                    rad_t = rad + p.margin + vb * (l + 1) * p.dt
                    if l == N - 1:
                        rad_t += p.backup_clearance
                    data[self._nfz_idx[j, l]] = n
                    lo[r] = rad_t + n @ (cx, cy)
                else:
                    data[self._nfz_idx[j, l]] = 0.0
                    lo[r] = -np.inf
        b = np.zeros(self.n_eq); b[:4] = self.A @ x0
        lo[:self.n_eq] = b; up[:self.n_eq] = b
        q = np.zeros(self.nz)
        for l in range(N):
            W = self.P if l == N - 1 else self.Q
            q[self._iX(l):self._iX(l) + 4] = -W @ ref[l]
        self.solver.update(q=q, l=lo, u=up, Ax=data)
        res = self.solver.solve(raise_error=False)
        ok = res.info.status in ("solved", "solved inaccurate") and res.x is not None
        if ok:
            U = res.x[self.n_eq:].reshape(N, 2)
            # accept only if the solution really satisfies the tightened constraints (safety check)
            X = self._predict(x0, U)[1:]
            ok = self._check(X, near)
        if ok:
            self.plan_u = U
            self.stats["solved"] += 1
            return U[0], dict(mode="mpc", plan=X)
        if len(self.plan_u) > 1:
            self.plan_u = self.plan_u[1:]
            self.stats["stored"] += 1
            return self.plan_u[0], dict(mode="stored", plan=None)
        self.plan_u = np.zeros((0, 2))
        self.stats["backup"] += 1
        return self.backup(x0, near), dict(mode="backup", plan=None)

    def _check(self, X, near, tol=1e-3):
        p = self.p
        for z in near:
            cx, cy, rad = z[:3]; vb = z[3] if len(z) > 3 else p.vbar
            d = np.hypot(X[:, 0] - cx, X[:, 1] - cy)
            need = rad + p.margin + vb * p.dt * np.arange(1, p.N + 1)
            need[-1] += p.backup_clearance
            if np.any(d < need - tol):
                return False
        return np.linalg.norm(X[-1, 2:]) < 1e-2
