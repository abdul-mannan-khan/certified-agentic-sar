"""R1.5 figure for a PX4 mission (AirSim SITL or HITL): states and control inputs over time.

  python hil/plot_px4_mission.py results/city_YYYYMMDD/city/best_of_7000 --uav 0 --out fig_px4.pdf [--city data/cityenviron_alt120_cell40.json]

Panels: (a) map: obstacles, NFZ tracks, flown paths of all UAVs; (b) position vs MPC reference;
(c) velocity vs set-point; (d) roll and pitch; (e) motor outputs; (f) altitude;
(g) clearance to the nearest moving NFZ and to the nearest building, with the margin.
"""
import argparse
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def rd(path):
    rows = list(csv.DictReader(open(path)))
    keys = rows[0].keys()
    return {k: np.array([float(r[k]) if r[k] not in ("", "nan") else np.nan for r in rows]) if k != "mode"
            else np.array([r[k] for r in rows]) for k in keys}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("run"); ap.add_argument("--uav", type=int, default=0)
    ap.add_argument("--city", default=None); ap.add_argument("--margin", type=float, default=10.0)
    ap.add_argument("--out", default="fig_px4.pdf")
    ap.add_argument("--win", default=None, help="t0,t1: zoom the time-series panels (b)-(g) to this mission-time window"); a = ap.parse_args()
    tr = rd(os.path.join(a.run, "trace.csv")); px = rd(os.path.join(a.run, "px4.csv"))
    world = json.load(open(os.path.join(a.run, "world.json"))); ep = json.load(open(os.path.join(a.run, "episode.json")))
    obst = json.load(open(a.city))["obstacles"] if a.city else []
    u = a.uav; m = tr["uav"] == u; k = px["k"] == u
    t = tr["t"][m]; tp = px["t_px4"][k] - px["t_px4"][k][0]
    # align the PX4 clock with mission time by matching the north velocity of the two logs
    vq = np.interp(t, t, tr["vx"][m]); best = (np.inf, 0.0)
    for off in np.arange(0.0, 120.0, 0.1):
        err = np.nanmean((np.interp(t, tp - off, px["vx"][k], left=np.nan, right=np.nan) - vq) ** 2)
        if err < best[0]:
            best = (err, off)
    tp = tp - best[1]
    plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.grid": True, "grid.alpha": .3})
    fig = plt.figure(figsize=(7.16, 7.0)); gs = fig.add_gridspec(4, 3)
    ax = fig.add_subplot(gs[:2, 0])
    for (x, y, r) in obst:
        ax.add_patch(plt.Circle((y, x), r, color="0.6", alpha=.5, lw=0))
    nt = world.get("nfz_trace", [])
    if nt:
        for j in range(len(nt[0][1])):
            P = np.array([z[1][j][:2] for z in nt if len(z[1]) > j])
            ax.plot(P[:, 1], P[:, 0], color="tab:red", lw=.8)
            ax.add_patch(plt.Circle((P[-1, 1], P[-1, 0]), nt[-1][1][j][2], color="tab:red", alpha=.2))
    for v in np.unique(tr["uav"]):
        mv = tr["uav"] == v
        ax.plot(tr["y"][mv], tr["x"][mv], lw=.8, label=f"UAV {int(v)}")
    ax.set_aspect("equal"); ax.set_xlabel("east (m)"); ax.set_ylabel("north (m)"); ax.legend(fontsize=6)
    ax.set_title("(a) paths; buildings, NFZs", fontsize=8)

    def panel(pos, title, series, ylabel, tt):
        axp = fig.add_subplot(pos)
        for y, lab, sty in series:
            axp.plot(tt, y, sty, lw=.8, label=lab)
        axp.set_title(title, fontsize=8); axp.set_ylabel(ylabel); axp.legend(fontsize=6, ncol=2)
        return axp
    P = [panel(gs[0, 1], "(b) position vs epoch waypoint", [(tr["x"][m], "north", "C0-"), (tr["rx"][m], "waypoint", "C0:"),
          (tr["y"][m], "east", "C1-"), (tr["ry"][m], "waypoint", "C1:")], "m", t)]
    P += [panel(gs[0, 2], "(c) velocity vs set-point", [(px["vx"][k], "v_N", "C0-"), (px["vx_sp"][k], "set-point", "C0:"),
          (px["vy"][k], "v_E", "C1-"), (px["vy_sp"][k], "set-point", "C1:")], "m/s", tp)]
    P += [panel(gs[1, 1], "(d) attitude", [(np.degrees(px["roll"][k]), "roll", "C0-"), (np.degrees(px["pitch"][k]), "pitch", "C1-")], "deg", tp)]
    P += [panel(gs[1, 2], "(e) motor outputs", [(px[f"u{j}"][k], f"m{j}", f"C{j-1}-") for j in range(1, 5)], "PWM / norm.", tp)]
    P += [panel(gs[2, 0], "(f) altitude", [(px["alt"][k], "altitude", "C2-")], "m", tp)]
    P += [panel(gs[2, 1], "(g) MPC acceleration command", [(tr["ax_cmd"][m], "a_N", "C0-"), (tr["ay_cmd"][m], "a_E", "C1-")], "m/s²", t)]
    # clearances
    cl_n = np.full(m.sum(), np.nan); cl_b = np.full(m.sum(), np.nan)
    if nt:
        tn = np.array([z[0] for z in nt])
        for q, (ti, xi, yi) in enumerate(zip(t, tr["x"][m], tr["y"][m])):
            zs = nt[int(np.clip(np.searchsorted(tn, ti), 0, len(nt) - 1))][1]
            if zs:
                cl_n[q] = min(np.hypot(xi - z[0], yi - z[1]) - z[2] for z in zs)
    if obst:
        O = np.array(obst)
        cl_b = np.min(np.hypot(tr["x"][m][:, None] - O[:, 0], tr["y"][m][:, None] - O[:, 1]) - O[:, 2], 1)
    axc = panel(gs[2, 2], "(h) clearance", [(cl_n, "moving NFZ", "C3-"), (cl_b, "building", "0.4")], "m", t)
    axc.axhline(a.margin, color="k", ls="--", lw=.7); axc.axhline(0, color="tab:red", lw=.8); axc.set_ylim(-5, 150)
    if a.win:
        w0, w1 = map(float, a.win.split(","))
        for axp in P:
            axp.set_xlim(w0, w1); axp.set_xlabel("time (s)")
    axm = fig.add_subplot(gs[3, :])
    modes = tr["mode"][m]
    for j, (lab, col) in enumerate((("mpc", "C2"), ("stored", "C1"), ("backup", "C3"))):
        sel = modes == lab
        axm.scatter(t[sel], np.full(sel.sum(), j), s=2, color=col, label=lab)
    axm.set_yticks([0, 1, 2]); axm.set_yticklabels(["QP solved", "stored plan", "backup"]); axm.set_xlabel("time (s)")
    axm.set_title("(i) execution-layer mode", fontsize=8)
    fig.tight_layout(); fig.savefig(a.out); print("saved", a.out)
    print(f"found {ep['found']}/{ep['n_survivors']}, incursion {ep['incursion']}, min NFZ clearance {ep['min_clearance']:.1f} m, "
          f"min building clearance {ep.get('min_static_clearance', float('nan')):.1f} m, MPC modes {ep['mpc_modes']}")


if __name__ == "__main__":
    main()
