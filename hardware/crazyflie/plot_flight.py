"""Figure for the response letter (R1.5): states and control inputs over time.

Usage: python plot_flight.py logs/cf_<run>.csv [more.csv ...] --out fig_cf.pdf
With several logs, the top-view panel overlays all runs and a summary table
(per run: completed?, RMS / max horizontal tracking error, min NFZ clearance)
is printed.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from plan_path import FloorSetup, make_plan  # noqa: E402


def load(path):
    rows = list(csv.DictReader(open(path)))
    out = {k: np.array([float(r[k]) if r[k] not in ("", "nan", "None") else np.nan for r in rows])
           for k in rows[0] if k != "phase"}
    out["phase"] = np.array([r["phase"] for r in rows])
    return out


def summary(d):
    m = d["phase"] == "route"
    e = np.hypot(d["x"][m] - d["tgt_x"][m], d["y"][m] - d["tgt_y"][m])  # vs instantaneous trajectory target
    return dict(rms=float(np.sqrt(np.nanmean(e ** 2))), emax=float(np.nanmax(e)),
                clear=float(np.nanmin(d["nfz_clear"][m])), dur=float(d["t"][-1]),
                done=bool(np.nanmax(d["wp"]) >= 12 and "estop" not in set(d["phase"])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--out", default="fig_cf.pdf")
    a = ap.parse_args()
    s = FloorSetup(); plan = make_plan(s)
    runs = [load(p) for p in a.logs]
    d = runs[0]
    plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.grid": True, "grid.alpha": .3})
    fig = plt.figure(figsize=(7.16, 6.2))
    gs = fig.add_gridspec(3, 3)
    ax = fig.add_subplot(gs[:2, 0])
    x0, y0, x1, y1 = s.nfz
    ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, color="tab:red", alpha=.25, label="NFZ"))
    ax.add_patch(plt.Rectangle((x0 - s.nfz_margin, y0 - s.nfz_margin), x1 - x0 + 2 * s.nfz_margin,
                               y1 - y0 + 2 * s.nfz_margin, fill=False, ls="--", ec="tab:red", lw=.8))
    rx = [p[0] for p in plan.route]; ry = [p[1] for p in plan.route]
    ax.plot(rx, ry, "k--", lw=.8, label="planned route")
    for i, r in enumerate(runs):
        m = r["phase"] == "route"
        ax.plot(r["x"][m], r["y"][m], lw=1, label="estimate" if i == 0 else None, alpha=.8)
    ax.plot(*zip(*plan.waypoints), "ko", ms=3, label="decisions")
    ax.set_xlim(-.1, s.size + .1); ax.set_ylim(-.1, s.size + .1); ax.set_aspect("equal")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_title("(a) top view", fontsize=8)
    ax.legend(fontsize=6, loc="lower right")
    t = d["t"]
    def panel(pos, title, series, ylabel):
        axp = fig.add_subplot(pos)
        for y, lab, sty in series:
            axp.plot(t, y, sty, lw=.9, label=lab)
        axp.set_title(title, fontsize=8); axp.set_ylabel(ylabel); axp.legend(fontsize=6, ncol=3)
        return axp
    panel(gs[0, 1], "(b) position vs reference", [(d["x"], "x", "C0-"), (d["tgt_x"], "x ref", "C0:"),
          (d["y"], "y", "C1-"), (d["tgt_y"], "y ref", "C1:"), (d["z"], "z", "C2-"), (d["tgt_z"], "z ref", "C2:")], "m")
    panel(gs[0, 2], "(c) velocity", [(d["vx"], "vx", "C0-"), (d["vy"], "vy", "C1-"), (d["vz"], "vz", "C2-")], "m/s")
    panel(gs[1, 1], "(d) attitude", [(d["roll"], "roll", "C0-"), (d["pitch"], "pitch", "C1-")], "deg")
    panel(gs[1, 2], "(e) motor commands", [(d["m1"], "m1", "C0-"), (d["m2"], "m2", "C1-"),
          (d["m3"], "m3", "C2-"), (d["m4"], "m4", "C3-")], "PWM")
    axr = panel(gs[2, :2], "(f) Multi-ranger distances", [(d["r_front"], "front", "C0-"), (d["r_back"], "back", "C1-"),
                (d["r_left"], "left", "C2-"), (d["r_right"], "right", "C3-"), (d["r_z"], "down", "C4-")], "m")
    axr.set_xlabel("time (s)"); axr.set_ylim(0, 2.2)
    axc = panel(gs[2, 2], "(g) NFZ clearance", [(d["nfz_clear"], "clearance", "C3-")], "m")
    axc.axhline(s.nfz_margin, color="k", ls="--", lw=.8); axc.axhline(0, color="tab:red", lw=1)
    axc.set_xlabel("time (s)")
    fig.tight_layout(); fig.savefig(a.out); print("saved", a.out)
    print("run | completed | RMS err (m) | max err (m) | min NFZ clearance (m) | duration (s)")
    for p, r in zip(a.logs, runs):
        q = summary(r)
        print(f"{os.path.basename(p)} | {q['done']} | {q['rms']:.3f} | {q['emax']:.3f} | {q['clear']:.3f} | {q['dur']:.1f}")


if __name__ == "__main__":
    main()
