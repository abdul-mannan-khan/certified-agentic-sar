"""Static obstacles for the AirSim CityEnviron missions, from a surveyed height map.

Every grid cell whose highest surface (with a small border) rises above `alt - clearance` becomes a
disc obstacle circumscribing the cell; the planner blocks it and the MPC treats it as a no-fly zone
with speed bound 0. Mission frame: x = north + offset, y = east + offset (AirSim NED relative to
PlayerStart), so the base (offset, offset) is the plaza spawn point.

  python hil/city_obstacles.py --heightmap <heightmap.npz> --alt 120 --clearance 20 --cell 40 --size 560 \
      --offset 20 --out data/cityenviron_alt120_cell40.json
The height map (fields h, valid, n0, e0, res; h = up-positive height, 1 m grid) is a downward depth
survey of the public AirSim 1.8.1 CityEnviron map.
"""
import argparse
import json

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--heightmap", required=True); ap.add_argument("--alt", type=float, default=120)
ap.add_argument("--clearance", type=float, default=20); ap.add_argument("--cell", type=float, default=40)
ap.add_argument("--size", type=float, default=560); ap.add_argument("--offset", type=float, default=20)
ap.add_argument("--border", type=float, default=5); ap.add_argument("--out", required=True)
a = ap.parse_args()
d = np.load(a.heightmap, allow_pickle=True)
H = np.where(d["valid"].astype(bool), d["h"], np.nan); n0, e0, res = float(d["n0"]), float(d["e0"]), float(d["res"])
nc = int(round(a.size / a.cell)); obs = []; ceil = np.zeros((nc, nc))
for i in range(nc):          # i: north index (mission x), j: east index (mission y)
    for j in range(nc):
        N0, E0 = i * a.cell - a.offset, j * a.cell - a.offset
        r0, c0 = int((N0 - a.border - n0) / res), int((E0 - a.border - e0) / res)
        k = int((a.cell + 2 * a.border) / res)
        blk = H[max(0, r0):r0 + k, max(0, c0):c0 + k]
        m = float(np.nanmax(blk)) if np.isfinite(blk).any() else 0.0
        ceil[i, j] = m
        if m > a.alt - a.clearance:
            obs.append([(i + 0.5) * a.cell, (j + 0.5) * a.cell, a.cell / np.sqrt(2)])
json.dump(dict(alt=a.alt, clearance=a.clearance, cell=a.cell, size=a.size, offset=a.offset,
               obstacles=obs, ceiling=ceil.round(1).tolist(), source="downward depth survey of AirSim 1.8.1 CityEnviron"),
          open(a.out, "w"))
print(len(obs), "obstacle cells of", nc * nc)
