"""Dedicated timing run (one process, nothing else on the core): planner per epoch and MPC per step.

  taskset -c 0 python experiments/timing.py --seeds 1000-1004 --out results/timing
"""
import argparse, json, os, platform, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from experiments.run_campaign import parse_seeds  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", default="1000-1004"); ap.add_argument("--out", required=True)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    from sar.sim import Mission, SimConfig
    import sar.nmpc as N
    mpc_t = []; orig = N.SafeMPC.step
    def timed(self, *k, **kw):
        t0 = time.perf_counter(); r = orig(self, *k, **kw); mpc_t.append(1e3 * (time.perf_counter() - t0)); return r
    N.SafeMPC.step = timed
    plan = []
    for s in parse_seeds(a.seeds):
        r = Mission(SimConfig(), s).run("best_of"); plan += r.plan_ms
    cpu = platform.processor() or open("/proc/cpuinfo").read().split("model name")[1].split("\n")[0].strip(": \t")
    q = lambda v: dict(p50=float(np.percentile(v, 50)), p95=float(np.percentile(v, 95)), max=float(np.max(v)), n=len(v))
    out = dict(cpu=cpu, plan_ms=q(plan), mpc_ms=q(mpc_t))
    json.dump(out, open(os.path.join(a.out, "timing.json"), "w"), indent=1); print(json.dumps(out))
