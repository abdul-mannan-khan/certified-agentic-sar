"""Calibration of the belief (Section VI, trust protocol).

At fixed mission times, every cell's posterior p_c (probability of an undetected survivor) is paired with the
truth (survivor present and not yet found). Reliability diagram, Brier score and ECE (15 equal-mass bins), for the
matched sensing model and for a planner whose model over- or under-states the true detection probability.

  python experiments/calibration.py --seeds 8000-8029 --out results/calibration
"""
import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from experiments.run_campaign import parse_seeds  # noqa: E402

CHECK = (300.0, 600.0, 900.0, 1200.0)


def one(args):
    seed, qm = args
    from sar.sim import Mission, SimConfig
    m = Mission(SimConfig(q_mismatch=qm), seed)
    snaps = []
    orig = m.sense
    def sense(i):                                  # snapshot at the first look after each check time
        hit = orig(i)
        while len(snaps) < len(CHECK) and m.t >= CHECK[len(snaps)] and i == m.cfg.n_uav - 1:
            snaps.append((m.p.copy(), (m.truth & ~m.found).copy()))
        return hit
    m.sense = sense
    m.run("best_of")
    P = np.concatenate([s[0] for s in snaps]) if snaps else np.zeros(0)
    Y = np.concatenate([s[1] for s in snaps]).astype(float) if snaps else np.zeros(0)
    return P, Y


def summary(P, Y, nb=15):
    o = np.argsort(P); bins = np.array_split(o, nb)
    rel = [(float(P[b].mean()), float(Y[b].mean()), len(b)) for b in bins if len(b)]
    ece = sum(n * abs(pm - ym) for pm, ym, n in rel) / len(P)
    return dict(brier=float(np.mean((P - Y) ** 2)), ece=float(ece), mean_p=float(P.mean()), mean_y=float(Y.mean()),
                n=int(len(P)), reliability=rel)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", default="8000-8029"); ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--out", required=True); a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    res = {}
    for qm in (1.0, 0.8, 1.2):
        with ProcessPoolExecutor(a.workers) as ex:
            out = list(ex.map(one, [(s, qm) for s in parse_seeds(a.seeds)]))
        P = np.concatenate([o[0] for o in out]); Y = np.concatenate([o[1] for o in out])
        res[str(qm)] = summary(P, Y); print(qm, {k: v for k, v in res[str(qm)].items() if k != "reliability"}, flush=True)
    json.dump(res, open(os.path.join(a.out, "calibration.json"), "w"), indent=1)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.grid": True, "grid.alpha": .3})
    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    lab = {"1.0": "matched model", "0.8": "true q = 0.8 × model (overconfident)", "1.2": "true q = 1.2 × model (underconfident)"}
    for k, r in res.items():
        R = np.array(r["reliability"]); ax.plot(R[:, 0], R[:, 1], "o-", ms=3, lw=.8, label=f"{lab[k]}: ECE {r['ece']:.4f}")
    mx = max(max(np.array(r["reliability"])[:, :2].max() for r in res.values()), 1e-3)
    ax.plot([0, mx], [0, mx], "k--", lw=.7); ax.set_xlabel("posterior p_c (bin mean)"); ax.set_ylabel("observed frequency")
    ax.legend(fontsize=6); fig.tight_layout(); fig.savefig(os.path.join(a.out, "fig_calibration.pdf"))
