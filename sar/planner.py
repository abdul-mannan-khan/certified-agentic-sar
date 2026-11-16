"""Certified tactical planning for one decision epoch.

Implements the results of Section IV of the paper:

* sequential greedy over a partition matroid (one waypoint per UAV from its
  reachable set) achieves at least 1/2 of the per-epoch optimum
  (Fisher, Nemhauser and Wolsey, 1978);
* "best-of": the Hungarian assignment is executed only if its value is at
  least that of greedy, so the 1/2 guarantee is kept;
* certification gate: a learned proposal is accepted only if its marginal
  gain is at least beta times the best marginal gain, which makes every step
  a beta-approximate greedy step and gives beta / (1 + beta) of the optimum
  (Goundan and Schulz, 2007; Singh et al., 2009);
* a data-dependent upper bound OPT <= F(S) + sum_i max_a gain(a | S);
* an epsilon-fallback that sends a UAV to the nearest high-residual cell when
  no reachable waypoint is worth at least epsilon (Theorem 4).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from .objective import EpochObjective


@dataclass
class EpochDecision:
    actions: list[int]
    value: float
    upper_bound: float
    method: str
    accepted: list[bool] = field(default_factory=list)  # gate outcome per UAV
    fallback: list[bool] = field(default_factory=list)

    @property
    def certified_ratio(self) -> float:
        return 1.0 if self.upper_bound <= 0 else self.value / self.upper_bound


def sequential_greedy(obj: EpochObjective, action_sets: Sequence[Sequence[int]],
                      order: Optional[Sequence[int]] = None) -> tuple[list[int], EpochObjective]:
    o = obj.copy()
    acts = [None] * len(action_sets)
    for i in (order if order is not None else range(len(action_sets))):
        cand = list(action_sets[i])
        g = o.gains(cand)
        acts[i] = cand[int(np.argmax(g))]
        o.add(acts[i])
    return acts, o


def hungarian(obj: EpochObjective, action_sets: Sequence[Sequence[int]]) -> tuple[list[int], EpochObjective]:
    """Assignment on independent (stand-alone) gains, as in the v1 system."""
    cells = sorted({a for A in action_sets for a in A})
    col = {c: j for j, c in enumerate(cells)}
    g = obj.gains(cells)
    W = np.full((len(action_sets), len(cells)), -1e9)
    for i, A in enumerate(action_sets):
        for a in A:
            W[i, col[a]] = g[col[a]]
    if len(cells) < len(action_sets):
        return None, None
    rows, cols = linear_sum_assignment(-W)
    if np.any(W[rows, cols] <= -1e8):  # some UAV got a cell outside its reachable set
        return None, None
    acts = [None] * len(action_sets)
    for i, j in zip(rows, cols):
        acts[i] = cells[j]
    o = obj.copy()
    for a in acts:
        o.add(a)
    return acts, o


def upper_bound(obj_after: EpochObjective, action_sets: Sequence[Sequence[int]]) -> float:
    """OPT <= F(S) + sum_i max_{a in A_i} gain(a | S) for any S (partition matroid)."""
    return obj_after.value() + sum(float(np.max(obj_after.gains(list(A)))) for A in action_sets)


def plan_epoch(obj: EpochObjective, action_sets: Sequence[Sequence[int]], *,
               proposer: Optional[Callable[[int, EpochObjective, list[int]], int]] = None,
               beta: float = 0.9, use_hungarian: bool = True,
               epsilon: float = 0.0,
               fallback_targets: Optional[Sequence[Optional[int]]] = None) -> EpochDecision:
    """Choose one waypoint per UAV with a certified per-epoch guarantee.

    proposer(i, objective_so_far, A_i) returns the learned policy's proposal
    for UAV i (e.g. the DQN's argmax over masked Q-values). If None, the
    planner runs plain greedy (with best-of Hungarian if enabled).

    fallback_targets[i], if given, is the next cell on the path to the nearest
    high-residual cell; it is used when the best gain of UAV i is < epsilon.
    """
    n = len(action_sets)
    base = obj.copy()

    if proposer is None:
        acts, o = sequential_greedy(base, action_sets)
        method = "greedy"
        if use_hungarian:
            h_acts, h_o = hungarian(base, action_sets)
            if h_o is not None and h_o.value() >= o.value():
                acts, o, method = h_acts, h_o, "hungarian"
        accepted = [True] * n
    else:
        o = base.copy()
        acts, accepted = [], []
        for i in range(n):
            A = list(action_sets[i])
            g = o.gains(A)
            best = float(np.max(g))
            prop = proposer(i, o, A)
            g_prop = o.gain(prop) if prop in A else -np.inf
            ok = g_prop >= beta * best
            a = prop if ok else A[int(np.argmax(g))]
            accepted.append(bool(ok))
            acts.append(a)
            o.add(a)
        method = "certified"

    fallback = [False] * n
    if epsilon > 0 and fallback_targets is not None:
        # Re-evaluate from the base objective in UAV order, replacing UAVs whose
        # best stand-alone gain is below epsilon by their fallback target.
        o2 = base.copy()
        new = []
        for i in range(n):
            A = list(action_sets[i])
            if fallback_targets[i] is not None and float(np.max(o2.gains(A))) < epsilon:
                new.append(int(fallback_targets[i]))
                fallback[i] = True
            else:
                new.append(acts[i])
            o2.add(new[-1])
        acts, o = new, o2

    return EpochDecision(actions=list(acts), value=o.value() - base.value() + 0.0,
                         upper_bound=upper_bound(o, action_sets) - base.value(),
                         method=method, accepted=accepted, fallback=fallback)
