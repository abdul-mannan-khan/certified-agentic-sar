"""Statistics, tables and figures for the revision (reads the campaign JSONL files).

  python experiments/analyze.py --results results/vast_20261117 --out results/vast_20261117/analysis

Reported per policy (main campaign, paired by mission seed):
  fraction of survivors found, expected detections / prior mass, time to find 50 % and 80 % of the survivors,
  certified ratio (mean, 5th percentile, minimum), planner and MPC compute times.
Tests: Wilcoxon signed-rank (paired by seed) against the reference policy, Holm correction, matched-pairs
rank-biserial effect size, bootstrap 95 % CIs of the mean. Safety: incursion rate with exact Clopper-Pearson 95 %.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np
from scipy import stats


def load(path):
    out = []
    for f in glob.glob(path):
        for line in open(f):
            try:
                j = json.loads(line)
            except Exception:
                continue
            if "error" not in j:
                out.append(j)
    return out


def cp_interval(k, n, a=0.05):
    lo = 0.0 if k == 0 else stats.beta.ppf(a / 2, k, n - k + 1)
    hi = 1.0 if k == n else stats.beta.ppf(1 - a / 2, k + 1, n - k)
    return lo, hi


def boot_ci(x, B=5000, seed=0):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) == 0:
        return (np.nan, np.nan)
    r = np.random.default_rng(seed); m = r.choice(x, (B, len(x))).mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def t_frac(ep, frac, t_max=1800.0):
    n = ep["n_survivors"]
    if n == 0:
        return np.nan
    k = int(np.ceil(frac * n)); ft = sorted(ep["found_times"])
    return ft[k - 1] if len(ft) >= k else t_max   # censored at the mission limit


def metrics(ep):
    n = max(1, ep["n_survivors"])
    return dict(found=ep["found"] / n, mass=ep["detected_mass"] / ep["prior_mass"],
                t50=t_frac(ep, .5), t80=t_frac(ep, .8),
                cert_mean=ep.get("cert_mean"), cert_p05=ep.get("cert_p05"), cert_min=ep.get("cert_min"),
                plan_p95=ep.get("plan_ms_p95"), mpc_p95=ep.get("mpc_ms_p95"),
                accept=ep.get("accept_rate"), fallback=ep.get("fallback_rate"))


def holm(p):
    p = np.asarray(p); o = np.argsort(p); adj = np.empty_like(p); run = 0
    for r, i in enumerate(o):
        run = max(run, (len(p) - r) * p[i]); adj[i] = min(1.0, run)
    return adj


def rank_biserial(d):
    d = np.asarray(d); d = d[d != 0]
    if len(d) == 0:
        return 0.0
    r = stats.rankdata(np.abs(d)); return float((r[d > 0].sum() - r[d < 0].sum()) / r.sum())


def main_table(eps, ref, out, label):
    by = defaultdict(dict)
    for e in eps:
        key = e["policy"] + (f"#{e['train_seed']}" if "train_seed" in e else "")
        by[key][e["seed"]] = metrics(e)
    # learned policies: average over training seeds per mission seed
    agg = defaultdict(lambda: defaultdict(list))
    for k, d in by.items():
        base = k.split("#")[0]
        for s, m in d.items():
            agg[base][s].append(m)
    pol = {p: {s: {k: np.nanmean([m[k] if m[k] is not None else np.nan for m in ms]) for k in ms[0]}
               for s, ms in d.items()} for p, d in agg.items()}
    rows, tests = [], []
    keys = ["found", "mass", "t50", "t80"]
    for p, d in sorted(pol.items()):
        row = dict(policy=p, n=len(d))
        for k in keys + ["cert_mean", "cert_p05", "cert_min", "plan_p95", "mpc_p95", "accept", "fallback"]:
            v = np.array([m[k] for m in d.values()], float)
            row[k] = float(np.nanmean(v)) if np.isfinite(v).any() else None
            if k in keys:
                row[k + "_ci"] = boot_ci(v)
            if k == "cert_min":
                row[k] = float(np.nanmin(v)) if np.isfinite(v).any() else None
        rows.append(row)
        if p != ref and ref in pol:
            common = sorted(set(d) & set(pol[ref]))
            for k in keys:
                a = np.array([pol[ref][s][k] for s in common]); b = np.array([d[s][k] for s in common])
                diff = a - b
                pv = stats.wilcoxon(a, b, zero_method="zsplit").pvalue if np.any(diff != 0) else 1.0
                tests.append(dict(metric=k, vs=p, n=len(common), mean_diff=float(np.mean(diff)), p=float(pv),
                                  r_rb=rank_biserial(diff)))
    if tests:
        adj = holm([t["p"] for t in tests])
        for t, a in zip(tests, adj):
            t["p_holm"] = float(a)
    json.dump(dict(rows=rows, tests=tests, reference=ref), open(os.path.join(out, f"{label}_table.json"), "w"), indent=1)
    lines = [f"## {label}: reference = {ref}", "",
             "| policy | n | found % | E[det]/prior | t50 (s) | t80 (s) | cert mean | cert p5 | cert min | plan p95 ms | MPC p95 ms | gate accept |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    f = lambda v, d=2: "–" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{d}f}"
    for r in rows:
        lines.append(f"| {r['policy']} | {r['n']} | {f(100*r['found'],1)} [{f(100*r['found_ci'][0],1)}, {f(100*r['found_ci'][1],1)}] | "
                     f"{f(r['mass'])} | {f(r['t50'],0)} | {f(r['t80'],0)} | {f(r['cert_mean'])} | {f(r['cert_p05'])} | {f(r['cert_min'])} | "
                     f"{f(r['plan_p95'],1)} | {f(r['mpc_p95'],2)} | {f(r['accept'])} |")
    lines += ["", "| metric | ref − policy | n | mean diff | p (Wilcoxon) | p (Holm) | rank-biserial r |", "|---|---|---|---|---|---|---|"]
    for t in tests:
        lines.append(f"| {t['metric']} | {ref} − {t['vs']} | {t['n']} | {t['mean_diff']:.3f} | {t['p']:.2g} | {t['p_holm']:.2g} | {t['r_rb']:.2f} |")
    return "\n".join(lines) + "\n"


def safety_block(name, eps):
    n = len(eps); k = sum(e["incursion"] for e in eps)
    lo, hi = cp_interval(k, n)
    mc = np.array([e["min_clearance"] for e in eps])
    modes = defaultdict(int)
    for e in eps:
        for m, v in e["mpc_modes"].items():
            modes[m] += v
    tot = sum(modes.values()) or 1
    a3 = sum(1 for e in eps if e.get("a3_violation_steps", 0) > 0)
    k3 = sum(1 for e in eps if e["incursion"] and e.get("a3_violation_steps", 0) > 0)
    name = f"{name} (A3 failed in {a3} ep., {k3} of the incursions)"
    return (f"| {name} | {n} | {k} | {100*k/n if n else 0:.2f} | [{100*lo:.2f}, {100*hi:.2f}] | {np.min(mc):.1f} | {np.percentile(mc, 5):.1f} | "
            f"{100*modes['stored']/tot:.3f} | {100*modes['backup']/tot:.4f} |"), dict(n=n, k=k, cp=(lo, hi), min_clear=float(np.min(mc)))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--results", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); R = a.results
    md = ["# Results summary (generated by experiments/analyze.py)", ""]
    main = load(f"{R}/main/*.jsonl")
    if main:
        ref = "certified" if any(e["policy"] == "certified" for e in main) else "best_of"
        md.append(main_table(main, ref, a.out, "main"))
    md += ["## Safety (no-fly-zone incursions)", "",
           "| campaign | episodes | incursions | rate % | 95 % CP | min clearance (m) | 5th pct clearance | stored-plan % | backup % |",
           "|---|---|---|---|---|---|---|---|---|"]
    saf = {}
    for name, pat in [("main (all policies)", f"{R}/main/*.jsonl"), ("safety", f"{R}/safety/safety.jsonl"),
                      ("pop-up (violates A1-A2)", f"{R}/popup/popup.jsonl"), ("misspecified q", f"{R}/mismatch/*.jsonl"),
                      ("2 / 8 UAVs", f"{R}/scale/*.jsonl")]:
        eps = load(pat)
        if eps:
            line, s = safety_block(name, eps); md.append(line); saf[name] = s
    pooled = [e for p in ("main/*.jsonl", "safety/safety.jsonl", "mismatch/*.jsonl", "scale/*.jsonl") for e in load(f"{R}/{p}")]
    if pooled:
        line, s = safety_block("all bounded-speed (pooled)", pooled); md.append(line); saf["pooled"] = s
    md.append("")
    for name in ("mismatch", "scale"):
        eps = load(f"{R}/{name}/*.jsonl")
        if eps:
            md += [f"## {name}", ""]
            groups = defaultdict(list)
            for e in eps:
                ov = e.get("overrides", {}); groups[(e["policy"], json.dumps(ov, sort_keys=True))].append(metrics(e))
            md += ["| policy | setting | n | found % | E[det]/prior | t50 | t80 |", "|---|---|---|---|---|---|---|"]
            for (p, ov), ms in sorted(groups.items()):
                g = lambda k: np.nanmean([m[k] for m in ms])
                md.append(f"| {p} | {ov} | {len(ms)} | {100*g('found'):.1f} | {g('mass'):.2f} | {g('t50'):.0f} | {g('t80'):.0f} |")
            md.append("")
    milp = load(f"{R}/milp/*.jsonl")
    if milp:
        r = np.array([m["ratio_vs_ub"] for m in milp]); rl = np.array([m["ratio_vs_lb"] for m in milp])
        gap = np.array([(m["ub"] - m["lb"]) / m["ub"] for m in milp])
        md += ["## Mission-level gap (MILP)", "",
               f"{len(milp)} instances. Planner / MILP upper bound: mean {r.mean():.3f}, worst {r.min():.3f}. "
               f"Planner / best MILP solution: mean {rl.mean():.3f}, worst {rl.min():.3f}. MILP bracket width: mean {100*gap.mean():.2f} %.", ""]
        json.dump(dict(n=len(milp), mean_ub=float(r.mean()), worst_ub=float(r.min()), mean_lb=float(rl.mean()), worst_lb=float(rl.min())),
                  open(os.path.join(a.out, "milp.json"), "w"))
    curves = sorted(glob.glob(f"{R}/ddqn/seed*/curve.json"))
    if curves:
        md += ["## DDQN training (pre-registered stopping rule)", "", "| seed | steps | final eval | best eval | stopped by rule |", "|---|---|---|---|---|"]
        tl = open(f"{R}/train.log").read() if os.path.exists(f"{R}/train.log") else ""
        for c in curves:
            j = json.load(open(c)); cv = j["curve"]; sd = j["train"]["seed"]
            md.append(f"| {sd} | {cv[-1]['step']} | {cv[-1]['eval']:.3f} | {max(x['eval'] for x in cv):.3f} | "
                      f"{'yes' if f'seed {sd}: stopping rule met' in tl else 'no (budget)'} |")
        md.append("")
    json.dump(saf, open(os.path.join(a.out, "safety.json"), "w"), indent=1, default=float)
    open(os.path.join(a.out, "SUMMARY.md"), "w").write("\n".join(md))
    print("\n".join(md))
    figures(R, a.out, main, curves)


def figures(R, out, main, curves):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.grid": True, "grid.alpha": .3})
    if main:
        fig, ax = plt.subplots(1, 2, figsize=(7.16, 2.6))
        pols = sorted({e["policy"] for e in main}); tgrid = np.linspace(0, 1800, 181)
        for p in pols:
            eps = [e for e in main if e["policy"] == p]
            F = np.array([[np.sum(np.array(e["found_times"]) <= t) / max(1, e["n_survivors"]) for t in tgrid] for e in eps])
            m = F.mean(0); lo, hi = np.percentile(F, [2.5, 97.5], 0) if len(F) > 1 else (m, m)
            se = F.std(0) / np.sqrt(len(F))
            ax[0].plot(tgrid, 100 * m, label=p); ax[0].fill_between(tgrid, 100 * (m - 1.96 * se), 100 * (m + 1.96 * se), alpha=.2)
            cm = [e.get("cert_mean") for e in eps if e.get("cert_mean") is not None]
            if cm:
                ax[1].hist(cm, bins=20, alpha=.5, label=p)
        ax[0].set_xlabel("time (s)"); ax[0].set_ylabel("survivors found (%)"); ax[0].legend(fontsize=6)
        ax[0].set_title("(a) detections over time, mean ± 95 % CI", fontsize=8)
        ax[1].axvline(0.5, color="k", ls="--", lw=.8); ax[1].set_xlabel("mean certified ratio F/U per mission")
        ax[1].set_title("(b) per-epoch certificate", fontsize=8); ax[1].legend(fontsize=6)
        fig.tight_layout(); fig.savefig(os.path.join(out, "fig_main.pdf"))
    if curves:
        fig, ax = plt.subplots(figsize=(3.5, 2.4))
        C = [json.load(open(c))["curve"] for c in curves]
        L = min(len(c) for c in C); S = np.array([[x["eval"] for x in c[:L]] for c in C]); st = [x["step"] for x in C[0][:L]]
        for c in C:
            ax.plot([x["step"] for x in c], [x["eval"] for x in c], lw=.5, alpha=.5)
        ax.plot(st, S.mean(0), "k", lw=1.2, label="mean of seeds"); ax.fill_between(st, S.min(0), S.max(0), alpha=.2, label="min–max over seeds")
        ax.set_xlabel("environment steps"); ax.set_ylabel("held-out expected detections"); ax.legend(fontsize=6)
        fig.tight_layout(); fig.savefig(os.path.join(out, "fig_learning.pdf"))


if __name__ == "__main__":
    main()
