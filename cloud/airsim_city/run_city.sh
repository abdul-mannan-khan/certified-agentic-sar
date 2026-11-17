#!/bin/bash
# ON THE VAST GPU BOX (image abdulmannan617/airsim-city-wine:1.8.1-wine11-dxvk3.1.1).
# PX4 SITL x4 in AirSim CityEnviron, certified planner + safe MPC over MAVLink. Resumable; writes /work/results/DONE.
set -u
cd /work/certified-agentic-sar; R=/work/results; mkdir -p $R/city
export PYTHONPATH=$PWD
PY=/work/venv/bin/python
log() { echo "$(date -u +%F_%T) $*" | tee -a $R/run.log; }
city-selftest --quick > $R/selftest.txt 2>&1; log "selftest: $(tail -1 $R/selftest.txt)"
city-settings --mode px4 --n 4 --clock 0.7 > /work/s.json
city-start /work/s.json >> $R/city_start.log 2>&1 || { log "city-start failed"; exit 1; }
grep -m1 -i "CityEnviron" /work/logs/city.log > $R/scene.txt 2>/dev/null
SP="0,-4 0,0 0,4 4,-4"     # city-settings spawn spots of UAV0..3 (N,E metres from PlayerStart)
while read pol seed; do
  [ -z "$pol" ] && continue
  out=$R/city/${pol}_$seed; [ -f $out/episode.json ] && continue
  for try in 1 2; do
    px4-sitl stop > /dev/null 2>&1
    $PY - <<'PYEOF' > /dev/null 2>&1
import airsim; c = airsim.MultirotorClient("127.0.0.1"); c.confirmConnection(); c.reset()
PYEOF
    px4-sitl start 4 > $R/px4_start.log 2>&1 < /dev/null
    sleep 20
    timeout 2400 $PY hil/run_mission_px4.py --city data/cityenviron_alt102_cell40_sub10.json --alt 120 --layer 6 --t-max 600 \
      --urls udpin:0.0.0.0:14800 udpin:0.0.0.0:14801 udpin:0.0.0.0:14802 udpin:0.0.0.0:14803 \
      --spawns $SP --policy $pol --seed $seed --out $out > $out.log 2>&1 && break
    log "$pol $seed try $try failed: $(tail -1 $out.log)"
    city-status >> $R/run.log 2>&1
  done
  mkdir -p $out/ulog && cp /work/px4_run/px4_*/log/*/*.ulg $out/ulog/ 2>/dev/null; for i in 0 1 2 3; do cp /work/px4_run/px4_$i/out.log $out/ulog/px4_$i.out.log 2>/dev/null; done
  log "$pol $seed: $(tail -1 $out.log)"
done < cloud/airsim_city/plan.txt
px4-sitl stop > /dev/null 2>&1; city-stop > /dev/null 2>&1
touch $R/DONE
