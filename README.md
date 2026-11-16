# Certified Agentic SAR

Code for *Hierarchical Agentic AI with Submodular Optimization Guarantees for UAV Search and Rescue Coverage Planning* (A. M. Khan, IEEE Access, under review).

Multi-UAV search planning has three layers:

- **Tactical layer.** Each UAV chooses one waypoint per epoch from the cells it can reach without entering a no-fly zone. Sequential greedy on the expected-detection objective guarantees at least 1/2 of the per-epoch optimum. A learned (DQN) proposal is executed only if its marginal gain is within a factor β of the greedy gain; this keeps a β/(1+β) guarantee. Every epoch also reports a data-dependent upper bound on the optimum.
- **Execution layer.** NMPC with a terminal cost and terminal set, and constraints tightened for no-fly zones moving at bounded speed.
- **Search model.** Range-dependent detection probability and a Bayesian posterior over survivor locations.

## Layout
```
sar/objective.py   detection model, epoch objective F_k, Bayesian update
sar/reachable.py   moving NFZs, constraint tightening, reachable waypoint sets
sar/planner.py     greedy / best-of Hungarian / certified DQN gate / upper bound / fallback
tests/             brute-force checks of every per-epoch guarantee
```

## Quick check
```bash
python -m pip install numpy scipy pytest
python -m pytest -q tests
```

## Licence
MIT. See `LICENSE`.
