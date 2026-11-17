#!/bin/bash
# LAPTOP. Rent the CPU box, start run_all.sh, then monitor: pull results every 20 min, destroy when DONE.
# Usage: cloud/launch.sh <offer_id> <hard_stop_hours>
set -u
D=$(cd "$(dirname "$0")" && pwd); cd $D; OFFER=$1; H=$2; LABEL=cas-access43870; LOG=$D/vast.log
L=${LOCAL_RESULTS:-$HOME/kfupm_ws/certified-agentic-sar/results/vast_$(date -u +%Y%m%d)}; mkdir -p $L
say() { echo "$(date -u +%F_%T) $*" | tee -a $LOG; }
ID=$(vastai create instance $OFFER --image python:3.12-bookworm --disk 40 --ssh --direct --label $LABEL \
  --onstart-cmd 'apt-get update -qq && apt-get install -y -qq openssh-server rsync git >/dev/null; mkdir -p /run/sshd; /usr/sbin/sshd; touch /root/.no_auto_tmux; touch /root/READY' \
  --raw 2>&1 | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['new_contract'] if d.get('success') else '')")
[ -z "$ID" ] && { say "create failed"; exit 1; }
echo $ID > $D/instance.txt; say "created $ID (offer $OFFER), hard stop $H h"
nohup bash -c "sleep $((H*3600)); vastai destroy instance $ID -y; echo \"\$(date -u +%F_%T) hard stop destroyed $ID\" >> $LOG" >/dev/null 2>&1 &
echo $! > $D/hardstop.pid
S=""
for t in $(seq 1 60); do
  read st ip port < <(vastai show instance $ID --raw 2>/dev/null | python3 -c "import json,sys;d=json.load(sys.stdin);print(d.get('actual_status'),d.get('public_ipaddr'),((d.get('ports') or {}).get('22/tcp') or [{}])[0].get('HostPort'))")
  if [ "$st" = running ] && [ "$port" != None ]; then
    S="ssh -p $port -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=15 -o BatchMode=yes -o LogLevel=ERROR -o ServerAliveInterval=30 root@$ip"
    $S 'test -f /root/READY' 2>/dev/null && break; S=""
  fi; sleep 20; done
[ -z "$S" ] && { say "never ready"; vastai destroy instance $ID -y; exit 1; }
echo "$S" > $D/ssh.txt; say "ready: $S"
$S 'mkdir -p /work && cd /work && git clone -q https://github.com/abdul-mannan-khan/certified-agentic-sar.git && python3 -m venv venv && . venv/bin/activate && pip -q install numpy scipy osqp matplotlib pytest pymavlink && pip -q install torch --index-url https://download.pytorch.org/whl/cpu && echo setup-ok' >> $LOG 2>&1
$S 'cd /work && NC_EFF=${NC_EFF:-64} setsid nohup bash certified-agentic-sar/cloud/run_all.sh > /work/run_all.out 2>&1 < /dev/null &'
say "run_all started"
RS="rsync -az -e \"${S% root@*}\" ${S##* }:/work/results/ $L/"
last=$(date +%s)
while true; do
  sleep 1200
  eval $RS 2>>$LOG && last=$(date +%s)
  say "pulled: $(tail -1 $L/run.log 2>/dev/null)"
  if [ -f $L/DONE ]; then
    eval $RS; n_local=$(find $L -type f | wc -l); n_box=$($S 'find /work/results -type f | wc -l')
    if [ "$n_local" = "$n_box" ]; then vastai destroy instance $ID -y >> $LOG; kill $(cat $D/hardstop.pid) 2>/dev/null; say "DONE: $n_local files verified, destroyed $ID"; exit 0; fi
  fi
  if [ $(( $(date +%s) - last )) -gt 5400 ]; then say "stall: no pull for 90 min, destroying"; vastai destroy instance $ID -y; exit 1; fi
done
