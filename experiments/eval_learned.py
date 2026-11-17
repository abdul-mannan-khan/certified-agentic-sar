"""Evaluate trained proposers in the full simulator, gated (certified) and ungated.

  python experiments/eval_learned.py --models results/ddqn --seeds 1000-1099 --workers 32 --out results/main
"""
import argparse
import glob
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from experiments.run_campaign import parse_seeds  # noqa: E402


def work(model, train_seed, policy, seed):
    import numpy as np
    from sar.learn import LearnedProposer
    from sar.sim import SimConfig, run_episode
    import torch; torch.set_num_threads(1)
    cfg = SimConfig(); r = run_episode(policy, seed, cfg, proposer=LearnedProposer(model, cfg))
    cr = np.array(r.pop("certified_ratio")); pm = np.array(r.pop("plan_ms"))
    r.update(train_seed=train_seed, cert_min=float(cr.min()), cert_mean=float(cr.mean()), cert_p05=float(np.percentile(cr, 5)),
             plan_ms_p50=float(np.median(pm)), plan_ms_p95=float(np.percentile(pm, 95)))
    return r


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--models", required=True); ap.add_argument("--seeds", required=True)
    ap.add_argument("--workers", type=int, default=os.cpu_count()); ap.add_argument("--out", required=True)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); path = os.path.join(a.out, "learned.jsonl")
    done = set()
    if os.path.exists(path):
        for l in open(path):
            j = json.loads(l); done.add((j.get("train_seed"), j["policy"], j["seed"]))
    jobs = []
    for m in sorted(glob.glob(os.path.join(a.models, "seed*", "best.pt"))):
        ts = int(os.path.basename(os.path.dirname(m))[4:])
        jobs += [(m, ts, p, s) for s in parse_seeds(a.seeds) for p in ("certified", "ungated") if (ts, p, s) not in done]
    print(len(jobs), "episodes", flush=True)
    with ProcessPoolExecutor(a.workers) as ex, open(path, "a") as f:
        futs = [ex.submit(work, *j) for j in jobs]
        for k, fu in enumerate(as_completed(futs), 1):
            try:
                f.write(json.dumps(fu.result(), default=float) + "\n")
            except Exception as e:
                f.write(json.dumps(dict(error=repr(e), policy="?", seed=-1)) + "\n")
            f.flush()
            if k % 50 == 0:
                print(k, "/", len(jobs), flush=True)
