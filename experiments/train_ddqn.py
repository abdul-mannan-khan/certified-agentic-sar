"""Train the DDQN proposer for several seeds in parallel (pre-registered stopping rule in sar/learn.py).

  TORCH_THREADS=2 python experiments/train_ddqn.py --seeds 0-4 --out results/ddqn
"""
import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def one(seed, out):
    from sar.learn import TrainConfig, train
    from sar.sim import SimConfig
    return train(TrainConfig(seed=seed), SimConfig(), os.path.join(out, f"seed{seed}"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", default="0-4"); ap.add_argument("--out", required=True)
    a = ap.parse_args(); lo, _, hi = a.seeds.partition("-"); seeds = range(int(lo), int(hi or lo) + 1)
    with ProcessPoolExecutor(len(seeds)) as ex:
        for s, c in zip(seeds, ex.map(one, seeds, [a.out] * len(seeds))):
            print("seed", s, "finished at step", c[-1]["step"] if c else None, flush=True)
