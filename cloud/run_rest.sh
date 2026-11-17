#!/bin/bash
# ON THE VAST BOX: re-run the campaigns after a code fix while training/MILP processes keep running.
set -u
cd /work/certified-agentic-sar; R=/work/results; . /work/venv/bin/activate
export PYTHONPATH=$PWD OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
NC=${NC_EFF:-64}; W=$((NC - 12))
log() { echo "$(date -u +%F_%T) $*" | tee -a $R/run.log; }
log "campaigns restarted after metric fix ($(git rev-parse --short HEAD))"
python experiments/run_campaign.py --name main --policies best_of greedy hungarian lawnmower random --seeds 1000-1099 --workers $W --out $R/main > $R/main.log 2>&1; log "main done"
python experiments/run_campaign.py --name safety --policies best_of --seeds 2000-2499 --workers $W --out $R/safety > $R/safety.log 2>&1; log "safety done"
python experiments/run_campaign.py --name popup --policies best_of --seeds 3000-3199 --set popup=True --workers $W --out $R/popup > $R/popup.log 2>&1; log "popup done"
for q in 0.8 1.2; do python experiments/run_campaign.py --name mm$q --policies best_of --seeds 4000-4099 --set q_mismatch=$q --workers $W --out $R/mismatch > $R/mm$q.log 2>&1; done; log "mismatch done"
for n in 2 8; do python experiments/run_campaign.py --name uav$n --policies best_of lawnmower --seeds 6000-6049 --set n_uav=$n --workers $W --out $R/scale > $R/uav$n.log 2>&1; done; log "scale done"
while pgrep -f milp_gap.py >/dev/null; do sleep 60; done; log "milp done"
while pgrep -f train_ddqn.py >/dev/null; do sleep 120; done; log "training done: $(grep -c 'stopping rule met' $R/train.log) of 5 seeds met the stopping rule"
python experiments/eval_learned.py --models $R/ddqn --seeds 1000-1099 --workers $((NC - 2)) --out $R/main > $R/eval.log 2>&1; log "learned eval done"
touch $R/DONE
