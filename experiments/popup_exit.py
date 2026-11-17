"""Pop-up stress test detail: for every incursion, did the zone appear around the UAV, and how long until it left?

  python experiments/popup_exit.py --seeds 3000-3199 --workers 24 --out results/popup_exit
"""
import argparse, json, os, sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from experiments.run_campaign import parse_seeds  # noqa: E402


def one(seed):
    from sar.sim import Mission, SimConfig
    m = Mission(SimConfig(popup=True), seed, record=True); m.run("best_of")
    nt = m.nfz_trace; tn = np.array([z[0] for z in nt])
    events = []
    inside = {}
    for (t, i, x, y, *_rest) in m.trace:
        zs = nt[min(len(nt) - 1, int(np.searchsorted(tn, t)))][1]
        c = min((np.hypot(x - z[0], y - z[1]) - z[2] for z in zs), default=np.inf)
        if c < 0 and i not in inside:
            # zone appeared around the UAV if the previous NFZ snapshot had fewer zones
            k = min(len(nt) - 1, int(np.searchsorted(tn, t)))
            appeared = k > 0 and len(nt[k - 1][1]) < len(zs) or k == 0
            inside[i] = (t, appeared, c)
        elif c >= 0 and i in inside:
            t0, ap, c0 = inside.pop(i)
            events.append(dict(seed=seed, uav=int(i), t_in=t0, exit_s=t - t0, appeared_on_uav=bool(ap), depth=float(c0)))
    for i, (t0, ap, c0) in inside.items():
        events.append(dict(seed=seed, uav=int(i), t_in=t0, exit_s=None, appeared_on_uav=bool(ap), depth=float(c0)))
    return events


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", default="3000-3199"); ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--out", required=True); a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    with ProcessPoolExecutor(a.workers) as ex:
        ev = [e for es in ex.map(one, parse_seeds(a.seeds)) for e in es]
    json.dump(ev, open(os.path.join(a.out, "popup_exit.json"), "w"), indent=1)
    ex_ = np.array([e["exit_s"] for e in ev if e["exit_s"] is not None])
    print(len(ev), "entries;", sum(e["appeared_on_uav"] for e in ev), "began when a zone appeared around the UAV;",
          sum(e["exit_s"] is None for e in ev), "not exited by mission end;",
          "exit time median %.1f s, 95th %.1f s, max %.1f s" % (np.median(ex_), np.percentile(ex_, 95), ex_.max()) if len(ex_) else "")
