"""Summarise an AirSim CityEnviron PX4 campaign: one row per mission plus per-policy totals.

  python hil/summarize_city.py results/city_YYYYMMDD/city --out results/city_YYYYMMDD/summary.json
"""
import argparse
import csv
import glob
import json
import os

import numpy as np


def min_separation(px4_csv):
    """Minimum 3-D distance between any two vehicles, matching samples by time (0.25 s bins)."""
    rows = list(csv.DictReader(open(px4_csv)))
    by = {}
    for r in rows:
        b = round(float(r["t_px4"]) * 4)
        by.setdefault(b, {})[int(float(r["k"]))] = (float(r["x"]), float(r["y"]), float(r["alt"]))
    best = np.inf
    for s in by.values():
        P = list(s.values())
        for i in range(len(P)):
            for j in range(i + 1, len(P)):
                best = min(best, float(np.linalg.norm(np.subtract(P[i], P[j]))))
    return best


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("root"); ap.add_argument("--out", default=None); a = ap.parse_args()
    out = []
    for d in sorted(glob.glob(os.path.join(a.root, "*_*"))):
        f = os.path.join(d, "episode.json")
        if not os.path.isdir(d) or not os.path.exists(f):
            continue
        e = json.load(open(f))
        ev = json.load(open(os.path.join(d, "events.json"))) if os.path.exists(os.path.join(d, "events.json")) else []
        row = dict(policy=e["policy"], seed=e["seed"], found=e["found"], n=e["n_survivors"],
                   incursion=bool(e["incursion"]), min_nfz=e["min_clearance"],
                   min_bld=e.get("min_static_clearance"), reached=e.get("reached_rate"),
                   a3_steps=e.get("a3_violation_steps", 0), modes=e["mpc_modes"], mpc_p95=e.get("mpc_ms_p95"),
                   t_end=e["t_end"], jumps=sum(1 for x in ev if x.get("type") == "position_jump"),
                   min_sep=min_separation(os.path.join(d, "px4.csv")))
        out.append(row)
        print(f"{row['policy']:10s} {row['seed']} found {row['found']:2d}/{row['n']:2d} inc {row['incursion']!s:5s} "
              f"nfz {row['min_nfz']:6.1f} bld {row['min_bld']:5.1f} A3 {row['a3_steps']:4d} jumps {row['jumps']} "
              f"sep {row['min_sep']:.2f} mpc95 {row['mpc_p95']:.0f} ms")
    tot = {}
    for p in sorted({r["policy"] for r in out}):
        R = [r for r in out if r["policy"] == p]
        nm = sum(r["n"] for r in R); fd = sum(r["found"] for r in R)
        tot[p] = dict(missions=len(R), found=fd, survivors=nm, frac=fd / nm,
                      all_found=sum(r["found"] == r["n"] for r in R), incursions=sum(r["incursion"] for r in R),
                      a3_missions=sum(r["a3_steps"] > 0 for r in R),
                      min_nfz_no_inc=min((r["min_nfz"] for r in R if not r["incursion"]), default=None),
                      min_bld=min(r["min_bld"] for r in R), min_sep=min(r["min_sep"] for r in R),
                      jumps=sum(r["jumps"] for r in R), mpc_p95_median=float(np.median([r["mpc_p95"] for r in R])))
        print(p, json.dumps(tot[p]))
    if a.out:
        json.dump(dict(missions=out, totals=tot), open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
