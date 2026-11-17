"""Monte-Carlo campaign over policies x seeds (resumable, parallel).

  python experiments/run_campaign.py --name main --policies best_of greedy hungarian lawnmower random \
      --seeds 1000-1099 --workers 32 --out results/main
  python experiments/run_campaign.py --name safety --policies best_of --seeds 2000-2499 --out results/safety
  python experiments/run_campaign.py --name popup --policies best_of --seeds 3000-3199 --set popup=True --out results/popup
  python experiments/run_campaign.py --name mismatch --policies best_of --seeds 4000-4099 --set q_mismatch=0.8 --out results/mm08

Seeds 1-3 were used for tuning only; evaluation seeds start at 1000.
Each episode is appended as one JSON line to <out>/<name>.jsonl; finished (policy, seed) pairs are skipped on restart.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def parse_seeds(s):
    out = []
    for part in s.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def parse_set(kvs):
    d = {}
    for kv in kvs or []:
        k, v = kv.split("=")
        d[k] = {"True": True, "False": False}.get(v, None)
        if d[k] is None:
            try:
                d[k] = int(v)
            except ValueError:
                d[k] = float(v)
    return d


def work(policy, seed, overrides):
    from sar.sim import SimConfig, run_episode
    t0 = time.time()
    r = run_episode(policy, seed, SimConfig(**overrides))
    r["wall_s"] = time.time() - t0
    r["overrides"] = overrides
    # keep files small: summarise per-epoch lists
    import numpy as np
    cr = np.array(r.pop("certified_ratio")); pm = np.array(r.pop("plan_ms"))
    r["cert_min"] = float(cr.min()) if len(cr) else None
    r["cert_mean"] = float(cr.mean()) if len(cr) else None
    r["cert_p05"] = float(np.percentile(cr, 5)) if len(cr) else None
    r["plan_ms_p50"] = float(np.median(pm)) if len(pm) else None
    r["plan_ms_p95"] = float(np.percentile(pm, 95)) if len(pm) else None
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--policies", nargs="+", required=True)
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--set", nargs="*")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, a.name + ".jsonl")
    done = set()
    if os.path.exists(path):
        for line in open(path):
            try:
                j = json.loads(line); done.add((j["policy"], j["seed"]))
            except Exception:
                pass
    ov = parse_set(a.set)
    jobs = [(p, s) for s in parse_seeds(a.seeds) for p in a.policies if (p, s) not in done]
    print(f"{a.name}: {len(jobs)} episodes to run ({len(done)} done), {a.workers} workers, overrides {ov}", flush=True)
    with ProcessPoolExecutor(a.workers) as ex, open(path, "a") as f:
        futs = {ex.submit(work, p, s, ov): (p, s) for p, s in jobs}
        for k, fu in enumerate(as_completed(futs), 1):
            try:
                f.write(json.dumps(fu.result(), default=float) + "\n"); f.flush()
            except Exception as e:  # record the failure, keep going
                p, s = futs[fu]
                f.write(json.dumps(dict(policy=p, seed=s, error=repr(e))) + "\n"); f.flush()
            if k % 20 == 0 or k == len(jobs):
                print(f"{a.name}: {k}/{len(jobs)}", flush=True)


if __name__ == "__main__":
    main()
