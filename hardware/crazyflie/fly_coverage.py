"""Indoor single-UAV demonstration: fly the certified coverage route on a
Crazyflie 2.1+ with Flow v2 and Multi-ranger decks, respecting a virtual NFZ.

This is a single-UAV indoor demonstration of the tactical planner and the
safety logic. It is not the full multi-UAV system.

Safety layers (any one lands or stops the drone):
  * keys:  SPACE or e = emergency stop (motors off), l = land, q = land and quit
  * software geofence: estimate outside the floor box + 0.25 m -> land;
    outside + 0.45 m -> emergency stop
  * virtual NFZ: estimated clearance < 0 -> land (recorded as a failure)
  * Multi-ranger: any horizontal range < 0.10 m -> land (the route passes the NFZ wall at ~0.17 m)
  * setpoints are clamped inside the floor box before they are sent
  * low battery (< 3.3 V under load) -> land

Run on the laptop:
  python fly_coverage.py --dry-run            # no radio: simulated drone, full logs
  python fly_coverage.py --uri radio://0/80/2M/E7E7E7E7E7 --run-id r1
"""
from __future__ import annotations

import argparse
import csv
import os
import select
import sys
import termios
import threading
import time
import tty

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from plan_path import FloorSetup, box_clearance, make_plan  # noqa: E402

LOG_FIELDS = ["t", "phase", "wp", "sp_x", "sp_y", "sp_z", "tgt_x", "tgt_y", "tgt_z", "x", "y", "z", "vx", "vy", "vz",
              "roll", "pitch", "yaw", "m1", "m2", "m3", "m4", "thrust",
              "r_front", "r_back", "r_left", "r_right", "r_up", "r_z", "vbat", "nfz_clear"]


class KeyWatch(threading.Thread):
    """Reads single keys from the terminal without Enter."""

    def __init__(self):
        super().__init__(daemon=True)
        self.cmd = None
        self._fd = sys.stdin.fileno() if sys.stdin.isatty() else None

    def run(self):
        if self._fd is None:
            return
        old = termios.tcgetattr(self._fd)
        try:
            tty.setcbreak(self._fd)
            while self.cmd != "quit":
                if select.select([sys.stdin], [], [], 0.05)[0]:
                    k = sys.stdin.read(1).lower()
                    if k in (" ", "e"):
                        self.cmd = "estop"
                    elif k == "l":
                        self.cmd = "land"
                    elif k == "q":
                        self.cmd = "quit"
        finally:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, old)


class State(dict):
    def __init__(self):
        super().__init__({k: float("nan") for k in LOG_FIELDS})


# --------------------------------------------------------------------------- drones
class SimDrone:
    """Kinematic stand-in used by --dry-run (first-order tracking + noise)."""

    def __init__(self, st: State, rate=50.0):
        self.st, self.dt = st, 1.0 / rate
        self.pos = np.zeros(3)
        self.vel = np.zeros(3)
        self.target, self.t_end = np.zeros(3), 0.0
        self.ref = np.zeros(3)
        self.rng = np.random.default_rng(0)
        self.alive = True
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while self.alive:
            step = self.target - self.ref
            nrm = np.linalg.norm(step)
            self.ref = self.target.copy() if nrm < 0.4 * self.dt else self.ref + step / nrm * 0.4 * self.dt
            v_des = np.clip((self.ref - self.pos) * 3.0, -0.6, 0.6)
            self.vel += (v_des - self.vel) * 0.25
            self.pos += self.vel * self.dt + self.rng.normal(0, 0.002, 3)
            s = self.st
            s.update(x=self.pos[0], y=self.pos[1], z=self.pos[2], vx=self.vel[0], vy=self.vel[1],
                     vz=self.vel[2], roll=-8 * self.vel[1], pitch=8 * self.vel[0], yaw=0.0,
                     m1=30000, m2=30000, m3=30000, m4=30000, thrust=36000, vbat=3.9,
                     tgt_x=self.ref[0], tgt_y=self.ref[1], tgt_z=self.ref[2],
                     r_front=2.0, r_back=2.0, r_left=2.0, r_right=2.0, r_up=2.0, r_z=self.pos[2])
            time.sleep(self.dt)

    def takeoff(self, h, d):
        self.target = np.array([self.pos[0], self.pos[1], h])

    def go_to(self, x, y, z, d):
        self.target = np.array([x, y, z])

    def land(self, d):
        self.target = np.array([self.pos[0], self.pos[1], 0.0])

    def estop(self):
        self.target = self.pos.copy(); self.target[2] = 0.0

    def close(self):
        self.alive = False


class CfDrone:
    """Crazyflie via cflib high-level commander, with four log blocks."""

    def __init__(self, uri: str, st: State, rate_ms=20):
        import cflib.crtp
        from cflib.crazyflie import Crazyflie
        from cflib.crazyflie.log import LogConfig
        from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
        cflib.crtp.init_drivers()
        self.st = st
        self.scf = SyncCrazyflie(uri, cf=Crazyflie(rw_cache="./cache"))
        self.scf.open_link()
        cf = self.cf = self.scf.cf
        self._check_decks()
        cf.param.set_value("stabilizer.estimator", "2")      # Kalman
        cf.param.set_value("commander.enHighLevel", "1")
        cf.param.set_value("kalman.resetEstimation", "1"); time.sleep(0.1)
        cf.param.set_value("kalman.resetEstimation", "0"); time.sleep(2.0)
        blocks = {
            "pos": [("stateEstimate.x", "float"), ("stateEstimate.y", "float"), ("stateEstimate.z", "float"),
                    ("stateEstimate.vx", "float"), ("stateEstimate.vy", "float"), ("stateEstimate.vz", "float")],
            "att": [("stabilizer.roll", "float"), ("stabilizer.pitch", "float"), ("stabilizer.yaw", "float"),
                    ("ctrltarget.x", "float"), ("ctrltarget.y", "float"), ("ctrltarget.z", "float")],
            "mot": [("motor.m1", "uint16_t"), ("motor.m2", "uint16_t"), ("motor.m3", "uint16_t"),
                    ("motor.m4", "uint16_t"), ("controller.cmd_thrust", "float"), ("pm.vbat", "float")],
            "rng": [("range.front", "uint16_t"), ("range.back", "uint16_t"), ("range.left", "uint16_t"),
                    ("range.right", "uint16_t"), ("range.up", "uint16_t"), ("range.zrange", "uint16_t")],
        }
        names = {"stateEstimate.x": "x", "stateEstimate.y": "y", "stateEstimate.z": "z",
                 "stateEstimate.vx": "vx", "stateEstimate.vy": "vy", "stateEstimate.vz": "vz",
                 "stabilizer.roll": "roll", "stabilizer.pitch": "pitch", "stabilizer.yaw": "yaw",
                 "ctrltarget.x": "tgt_x", "ctrltarget.y": "tgt_y", "ctrltarget.z": "tgt_z",
                 "motor.m1": "m1", "motor.m2": "m2", "motor.m3": "m3", "motor.m4": "m4",
                 "controller.cmd_thrust": "thrust", "pm.vbat": "vbat",
                 "range.front": "r_front", "range.back": "r_back", "range.left": "r_left",
                 "range.right": "r_right", "range.up": "r_up", "range.zrange": "r_z"}
        self.logs = []
        for name, vars_ in blocks.items():
            lc = LogConfig(name=name, period_in_ms=rate_ms)
            for v, typ in vars_:
                lc.add_variable(v, typ)
            def cb(ts, data, conf, names=names):
                for k, v in data.items():
                    key = names[k]
                    if key.startswith("r_"):
                        v = v / 1000.0  # mm -> m
                    self.st[key] = float(v)
            lc.data_received_cb.add_callback(cb)
            cf.log.add_config(lc)
            lc.start()
            self.logs.append(lc)
        try:  # newer firmware may require arming; harmless otherwise
            cf.platform.send_arming_request(True); time.sleep(0.5)
        except Exception:
            pass
        self.hl = cf.high_level_commander

    def _check_decks(self):
        for p in ("deck.bcFlow2", "deck.bcMultiranger"):
            if int(self.cf.param.get_value(p)) != 1:
                raise RuntimeError(f"{p} not detected: check the deck is fitted")

    def takeoff(self, h, d): self.hl.takeoff(h, d)
    def go_to(self, x, y, z, d): self.hl.go_to(x, y, z, 0.0, d)
    def land(self, d): self.hl.land(0.0, d)

    def estop(self):
        try:
            self.hl.stop()
        finally:
            self.cf.commander.send_stop_setpoint()

    def close(self):
        for lc in self.logs:
            try:
                lc.stop()
            except Exception:
                pass
        self.scf.close_link()


# --------------------------------------------------------------------------- mission
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uri", default="radio://0/80/2M/E7E7E7E7E7")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--run-id", default=time.strftime("%Y%m%d_%H%M%S"))
    ap.add_argument("--height", type=float, default=0.5)
    ap.add_argument("--speed", type=float, default=0.4, help="flight speed m/s (<= 0.5)")
    ap.add_argument("--epochs", type=int, default=13)
    ap.add_argument("--hover-only", action="store_true", help="take off, hover 10 s, land")
    args = ap.parse_args()
    assert 0.2 <= args.height <= 0.8 and 0.1 <= args.speed <= 0.5

    setup = FloorSetup(epochs=args.epochs)
    plan = make_plan(setup)
    sx, sy = setup.start
    # The Flow deck estimate starts at (0,0) on the take-off spot: grid -> estimator frame.
    to_est = lambda x, y: (x - sx, y - sy)
    fence = (-0.25, -0.25, setup.size + 0.25, setup.size + 0.25)   # grid frame
    hard = (-0.45, -0.45, setup.size + 0.45, setup.size + 0.45)

    os.makedirs(os.path.join(os.path.dirname(__file__), "logs"), exist_ok=True)
    path = os.path.join(os.path.dirname(__file__), "logs", f"cf_{args.run_id}{'_dry' if args.dry_run else ''}.csv")
    st = State()
    drone = SimDrone(st) if args.dry_run else CfDrone(args.uri, st)
    keys = KeyWatch(); keys.start()
    sp = [0.0, 0.0, 0.0]
    status = {"phase": "init", "wp": 0, "abort": None}
    t0 = time.time()
    f = open(path, "w", newline="")
    w = csv.DictWriter(f, fieldnames=LOG_FIELDS); w.writeheader()

    def logger():
        while status["phase"] != "done":
            gx, gy = st["x"] + sx, st["y"] + sy          # back to grid frame
            row = dict(st); row.update(t=round(time.time() - t0, 3), phase=status["phase"], wp=status["wp"],
                                       sp_x=sp[0] + sx, sp_y=sp[1] + sy, sp_z=sp[2], x=gx, y=gy,
                                       tgt_x=st["tgt_x"] + sx, tgt_y=st["tgt_y"] + sy, tgt_z=st["tgt_z"],
                                       nfz_clear=box_clearance(gx, gy, setup.nfz) if np.isfinite(gx) else float("nan"))
            w.writerow({k: row.get(k) for k in LOG_FIELDS})
            # safety monitor
            if np.isfinite(gx) and status["phase"] in ("takeoff", "route", "hover"):
                if not (hard[0] <= gx <= hard[2] and hard[1] <= gy <= hard[3]):
                    status["abort"] = "estop: outside hard fence"
                elif not (fence[0] <= gx <= fence[2] and fence[1] <= gy <= fence[3]):
                    status["abort"] = status["abort"] or "land: outside geofence"
                elif box_clearance(gx, gy, setup.nfz) < 0:
                    status["abort"] = status["abort"] or "land: entered NFZ"
                elif status["phase"] == "route" and min(st["r_front"], st["r_back"], st["r_left"], st["r_right"]) < 0.10:
                    status["abort"] = status["abort"] or "land: obstacle closer than 0.10 m"
                elif st["vbat"] < 3.3:
                    status["abort"] = status["abort"] or "land: low battery"
            if keys.cmd == "estop":
                status["abort"] = "estop: key"
            elif keys.cmd in ("land", "quit"):
                status["abort"] = status["abort"] or "land: key"
            time.sleep(0.02)

    threading.Thread(target=logger, daemon=True).start()

    def wait(seconds):
        end = time.time() + seconds
        while time.time() < end:
            if status["abort"]:
                return False
            time.sleep(0.02)
        return True

    def send(x, y, z, d):
        x = min(max(x, 0.05), setup.size - 0.05); y = min(max(y, 0.05), setup.size - 0.05)  # clamp in grid frame
        ex, ey = to_est(x, y); sp[:] = [ex, ey, z]
        drone.go_to(ex, ey, z, d)

    ok = True
    try:
        print(f"Logging to {path}\nKeys: SPACE/e = EMERGENCY STOP, l = land, q = land+quit")
        status["phase"] = "takeoff"; sp[:] = [0.0, 0.0, args.height]
        drone.takeoff(args.height, 2.0)
        ok = wait(3.0)
        if ok and args.hover_only:
            status["phase"] = "hover"; ok = wait(10.0)
        elif ok:
            status["phase"] = "route"
            prev = plan.route[0]
            for x, y, k in plan.route[1:]:
                status["wp"] = k
                d = max(0.8, float(np.hypot(x - prev[0], y - prev[1])) / args.speed)
                send(x, y, args.height, d)
                if not wait(d + 0.3):
                    ok = False; break
                prev = (x, y, k)
    finally:
        reason = status["abort"]
        if reason and reason.startswith("estop"):
            status["phase"] = "estop"; drone.estop(); print("EMERGENCY STOP:", reason)
        else:
            status["phase"] = "land"
            if reason:
                print("Landing:", reason)
            drone.land(2.5); time.sleep(3.0)
        try:
            drone.estop() if not args.dry_run else None   # motors off after landing
        except Exception:
            pass
        status["phase"] = "done"; time.sleep(0.1)
        keys.cmd = "quit"; drone.close(); f.close()
        print("RESULT:", "completed" if ok and not reason else f"stopped ({reason})", "| log:", path)


if __name__ == "__main__":
    main()
