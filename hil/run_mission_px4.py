"""Run one certified-planner mission with PX4 vehicles (SITL in AirSim, or HITL).

The NFZs, survivors and sensing stay virtual (as in the internal simulator);
the vehicles are real PX4 flight stacks. Virtual teammates (internal model)
can be added, e.g. for the X500 HITL run: 1 Pixhawk + 2 virtual UAVs.

Examples
  # local smoke test against PX4 SITL (SIH), 1 PX4 + 2 virtual UAVs
  python hil/run_mission_px4.py --urls udpin:0.0.0.0:14540 --n-virtual 2 --size 600 --t-max 240 --out runs/sitl_smoke
  # Vast: 4 PX4 SITL vehicles in AirSim CityEnviron
  python hil/run_mission_px4.py --urls udpin:0.0.0.0:14800 udpin:0.0.0.0:14801 udpin:0.0.0.0:14802 udpin:0.0.0.0:14803 ...
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sar.plants import MixedPlant, ModelPlant, PX4Plant  # noqa: E402
from sar.sim import Mission, SimConfig, city_config  # noqa: E402
from dataclasses import asdict  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--urls", nargs="+", required=True)
    ap.add_argument("--n-virtual", type=int, default=0)
    ap.add_argument("--size", type=float, default=800.0)
    ap.add_argument("--t-max", type=float, default=600.0)
    ap.add_argument("--endurance", type=float, default=1e9)
    ap.add_argument("--alt", type=float, default=40.0)
    ap.add_argument("--n-nfz", type=int, default=3)
    ap.add_argument("--policy", default="best_of")
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--spacing", type=float, default=15.0, help="spawn spacing along y (m)")
    ap.add_argument("--city", default=None, help="obstacle json from hil/city_obstacles.py (AirSim CityEnviron)")
    ap.add_argument("--spawns", nargs="*", default=None, help="vehicle spawn offsets 'N,E' (m) relative to PlayerStart")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    n_px4 = len(a.urls); n = n_px4 + a.n_virtual
    if a.city:
        cfg = city_config(a.city, n_uav=n, t_max=a.t_max)
    else:
        cfg = SimConfig(size=a.size, n_uav=n, t_max=a.t_max, endurance=a.endurance, n_nfz=a.n_nfz,
                        nfz_r=(40.0, 80.0), n_survivor_expected=6.0, base=(25.0, 25.0))
    base = np.array(cfg.base)
    if a.spawns:
        homes = [base + np.array(list(map(float, sp.split(",")))) for sp in a.spawns]
    else:
        homes = [base + (0, a.spacing * i) for i in range(n)]
    px4_log = []
    px4 = PX4Plant(a.urls, homes[:n_px4], alt=a.alt, log=px4_log, dt=cfg.mpc.dt)
    plants = [px4]
    if a.n_virtual:
        x0 = np.array([[*homes[n_px4 + j], 0, 0] for j in range(a.n_virtual)])
        plants.append(ModelPlant(x0, cfg.mpc.dt, cfg.acc_lag, cfg.gust, np.random.default_rng(a.seed + 7)))
    plant = MixedPlant(plants)
    print("arming and climbing to", a.alt, "m", flush=True)
    px4.arm_takeoff()
    t0 = time.time()
    m = Mission(cfg, a.seed, plant=plant, record=True)
    try:
        res = m.run(a.policy)
    finally:
        print("landing", flush=True)
        px4.land_disarm()
    r = asdict(res); r.update(wall_s=time.time() - t0, urls=a.urls, n_virtual=a.n_virtual, alt=a.alt)
    json.dump(r, open(os.path.join(a.out, "episode.json"), "w"), indent=1, default=float)
    with open(os.path.join(a.out, "trace.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["t", "uav", "x", "y", "vx", "vy", "ax_cmd", "ay_cmd", "rx", "ry", "rvx", "rvy", "mode"])
        w.writerows(m.trace)
    with open(os.path.join(a.out, "px4.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["t_px4", "k", "x", "y", "vx", "vy", "alt", "roll", "pitch", "yaw",
                                       "u1", "u2", "u3", "u4", "vx_sp", "vy_sp", "ax_ff", "ay_ff"])
        w.writerows(px4_log)
    json.dump(px4.events, open(os.path.join(a.out, "events.json"), "w"), indent=1, default=float)
    nfz = [(z.x, z.y, z.r) for z in m.nfz]
    json.dump(dict(nfz_final=nfz, nfz_trace=m.nfz_trace[::5], truth=np.flatnonzero(m.truth).tolist(), found=np.flatnonzero(m.found).tolist()),
              open(os.path.join(a.out, "world.json"), "w"))
    print(f"done: found {res.found}/{res.n_survivors}, incursion {res.incursion}, min clearance {res.min_clearance:.1f} m, "
          f"MPC modes {res.mpc_modes}, wall {r['wall_s']:.0f} s", flush=True)


if __name__ == "__main__":
    main()
