"""Plants that the mission loop can drive (same planner and MPC code for all).

ModelPlant   double integrator with first-order acceleration lag and bounded
             gusts (fast Monte-Carlo campaigns).
PX4Plant     one PX4 vehicle over MAVLink in offboard mode: PX4 SITL in AirSim
             CityEnviron (Vast campaign) or the X500's Pixhawk 6C in HITL with
             AirSim (bench). The MPC acceleration is sent as a velocity setpoint
             (v + a dt) with acceleration feed-forward; PX4 holds altitude and
             attitude. Time is PX4's own clock, so lockstep simulation and real
             time are handled the same way.
MixedPlant   combines plants, e.g. one HITL Pixhawk plus virtual teammates.

Frames: the mission plane (x, y) maps to local NED as north = x - x_home,
east = y - y_home; altitude is held at `alt` metres.
"""
from __future__ import annotations

import time

import numpy as np


class ModelPlant:
    def __init__(self, x0, dt, lag, gust, rng):
        self.x = np.array(x0, float); self.acc = np.zeros((len(x0), 2))
        self.dt, self.lag, self.gust, self.rng = dt, lag, gust, rng
        self.cmd = np.zeros((len(x0), 2))

    def states(self):
        return self.x.copy()

    def apply(self, i, a, x_i=None, plan_next=None):
        self.cmd[i] = a

    def advance(self, dt):
        for i in range(len(self.x)):
            g = self.rng.normal(0, 1, 2); g = self.gust * g / max(1.0, np.linalg.norm(g))
            self.acc[i] += (self.cmd[i] - self.acc[i]) * min(1.0, dt / self.lag)
            ae = self.acc[i] + g
            self.x[i, :2] += self.x[i, 2:] * dt + 0.5 * ae * dt ** 2
            self.x[i, 2:] += ae * dt


class PX4Plant:
    """One or more PX4 vehicles over MAVLink (pymavlink)."""


    def __init__(self, urls, home_xy, alt=40.0, log=None, timeout=10.0, dt=0.2):
        from pymavlink import mavutil
        self.mavutil = mavutil
        self.links = [mavutil.mavlink_connection(u, source_system=250 + k) for k, u in enumerate(urls)]
        self.home = np.array(home_xy, float).reshape(len(urls), 2)
        # alt: one altitude for all, or one per vehicle (altitude layers keep teammates vertically separated)
        self.alts = np.broadcast_to(np.asarray(alt, float), (len(urls),)).copy()
        self.alt = float(np.min(self.alts))
        self.log, self.timeout, self.dt = log, timeout, dt
        self.x = np.zeros((len(urls), 4)); self.t_boot = np.zeros(len(urls))
        self.att = np.zeros((len(urls), 3)); self.act = np.full((len(urls), 4), np.nan)
        self.events = []
        for L in self.links:
            if L.wait_heartbeat(timeout=60) is None:
                raise RuntimeError("no heartbeat on " + str(L))
        self.cmd = np.zeros((len(urls), 4))      # vx, vy, ax, ay
        self.cmd_pos = [None] * len(urls)
        mav = mavutil.mavlink
        for L in self.links:                     # ask for the streams the R1.5 figure needs, at 20 Hz
            for msg_id in (mav.MAVLINK_MSG_ID_LOCAL_POSITION_NED, mav.MAVLINK_MSG_ID_ATTITUDE, mav.MAVLINK_MSG_ID_SERVO_OUTPUT_RAW):
                L.mav.command_long_send(L.target_system, L.target_component, mav.MAV_CMD_SET_MESSAGE_INTERVAL, 0,
                                        msg_id, 50_000, 0, 0, 0, 0, 0)
        self._poll(0.5)

    # --- telemetry ---
    def _poll(self, dur=0.0):
        end = time.time() + dur
        while True:
            for k, L in enumerate(self.links):
                while (m := L.recv_match(blocking=False)) is not None:
                    t = m.get_type()
                    if t == "LOCAL_POSITION_NED":
                        new = np.array([m.x + self.home[k, 0], m.y + self.home[k, 1], m.vx, m.vy])
                        tb = m.time_boot_ms / 1e3
                        if self.t_boot[k] > 0 and tb > self.t_boot[k]:
                            jump = np.linalg.norm(new[:2] - self.x[k, :2] - self.x[k, 2:] * (tb - self.t_boot[k]))
                            if jump > 3.0:                      # estimator jump: record it as an event
                                self.events.append(dict(t=tb, k=k, type="position_jump", size_m=float(jump)))
                        self.x[k] = new
                        self.t_boot[k] = tb
                        self.z = getattr(self, "z", np.zeros(len(self.links))); self.z[k] = -m.z
                    elif t == "STATUSTEXT":
                        self.events.append(dict(t=float(self.t_boot[k]), k=k, type="statustext", text=m.text))
                    elif t == "ATTITUDE":
                        self.att[k] = [m.roll, m.pitch, m.yaw]
                    elif t == "HIL_ACTUATOR_CONTROLS":          # HITL: normalised motor outputs
                        self.act[k] = m.controls[:4]
                    elif t == "SERVO_OUTPUT_RAW":               # SITL: motor PWM (us)
                        self.act[k] = [m.servo1_raw, m.servo2_raw, m.servo3_raw, m.servo4_raw]
            if time.time() >= end:
                return
            time.sleep(0.002)

    def states(self):
        return self.x.copy()

    # --- commands ---
    def _setpoint(self, k):
        L = self.links[k]
        vx, vy, ax, ay = self.cmd[k]
        if self.cmd_pos[k] is not None:
            # full trajectory set-point from the MPC plan: position, velocity and acceleration feed-forward
            n, e = self.cmd_pos[k] - self.home[k]
            mask = (1 << 5) | (1 << 8) | (1 << 10) | (1 << 11)
        else:
            # velocity + acceleration feed-forward only (stored plan / backup law)
            n = e = 0.0
            mask = (1 << 0) | (1 << 1) | (1 << 5) | (1 << 8) | (1 << 10) | (1 << 11)
        L.mav.set_position_target_local_ned_send(0, L.target_system, L.target_component,
                                                 self.mavutil.mavlink.MAV_FRAME_LOCAL_NED, mask,
                                                 n, e, -self.alts[k], vx, vy, 0, ax, ay, 0, 0, 0)

    def apply(self, i, a, x_i, plan_next=None):
        """plan_next: MPC-predicted state [px, py, vx, vy] one step ahead (None in stored/backup modes).
        Without a plan, the velocity set-point is integrated from the PREVIOUS SET-POINT, not from the measured
        velocity: rebuilding it from the measurement each step cancels the acceleration when the autopilot
        responds with a delay longer than one step."""
        a = np.asarray(a, float)
        if plan_next is not None:
            self.cmd_pos[i] = np.asarray(plan_next[:2], float)
            v = np.asarray(plan_next[2:4], float)
        else:
            self.cmd_pos[i] = None
            v = self.cmd[i, :2] + a * self.dt
            if np.linalg.norm(v - np.asarray(x_i[2:])) > 3.0:          # re-anchor if far from the real velocity
                v = np.asarray(x_i[2:]) + a * self.dt
        self.cmd[i] = [v[0], v[1], a[0], a[1]]

    def advance(self, dt):
        """Stream setpoints until vehicle 0's clock has advanced by dt (lockstep-safe)."""
        t0 = self.t_boot[0]; wall = time.time()
        while self.t_boot[0] < t0 + dt:
            for k in range(len(self.links)):
                self._setpoint(k)
            self._poll(0.02)
            if time.time() - wall > self.timeout:
                raise TimeoutError("PX4 clock did not advance")
        if self.log is not None:
            for k in range(len(self.links)):
                self.log.append((self.t_boot[k], k, *self.x[k], getattr(self, "z", [0] * 9)[k], *self.att[k], *self.act[k], *self.cmd[k]))

    # --- mission start / end ---
    def wait_estimate_settled(self, tol=0.3, window=10.0, max_wait=180.0):
        """On the ground, wait until every vehicle's position estimate moves less than tol over `window` s."""
        t0 = time.time(); hist = []
        while time.time() - t0 < max_wait:
            self._poll(0.5); hist.append((time.time(), self.x[:, :2].copy()))
            old = [h for h in hist if h[0] <= time.time() - window]
            if old and np.max(np.linalg.norm(self.x[:, :2] - old[-1][1], axis=1)) < tol:
                return time.time() - t0
        return None

    def arm_takeoff(self, climb_s=120.0):
        mav = self.mavutil.mavlink
        settled = self.wait_estimate_settled()
        self.events.append(dict(t=float(self.t_boot[0]), k=-1, type="estimate_settled", after_s=settled))
        self.cmd[:] = 0
        for _ in range(40):                      # PX4 needs setpoints before it accepts offboard
            for k in range(len(self.links)):
                self._setpoint(k)
            self._poll(0.05)
        for L in self.links:
            L.mav.command_long_send(L.target_system, L.target_component, mav.MAV_CMD_DO_SET_MODE, 0,
                                    1, 6, 0, 0, 0, 0, 0)                       # custom mode: offboard
            L.mav.command_long_send(L.target_system, L.target_component, mav.MAV_CMD_COMPONENT_ARM_DISARM, 0,
                                    1, 0, 0, 0, 0, 0, 0)
        t_end = time.time() + climb_s
        while time.time() < t_end:
            for k in range(len(self.links)):
                self._setpoint(k)
            self._poll(0.05)
            if np.all(getattr(self, "z", np.zeros(1)) > self.alts - 2):
                break

    def land_disarm(self):
        mav = self.mavutil.mavlink
        for L in self.links:
            L.mav.command_long_send(L.target_system, L.target_component, mav.MAV_CMD_NAV_LAND, 0, 0, 0, 0, 0, 0, 0, 0)
        t_end = time.time() + 90
        while time.time() < t_end and np.any(getattr(self, "z", np.zeros(1)) > 0.3):
            self._poll(0.2)
        for L in self.links:
            L.mav.command_long_send(L.target_system, L.target_component, mav.MAV_CMD_COMPONENT_ARM_DISARM, 0,
                                    0, 0, 0, 0, 0, 0, 0)
            L.mav.command_long_send(L.target_system, L.target_component, mav.MAV_CMD_DO_SET_MODE, 0,
                                    1, 4, 3, 0, 0, 0, 0)                       # Hold (handover rule 2)


class MixedPlant:
    """Concatenate plants: vehicles 0..n0-1 from the first plant, and so on.
    advance() advances the first (real-time / lockstep) plant first, then the rest."""

    def __init__(self, plants):
        self.plants = plants
        self.index = [(p, j) for p in plants for j in range(len(p.states()))]

    def states(self):
        return np.vstack([p.states() for p in self.plants])

    def apply(self, i, a, x_i, plan_next=None):
        p, j = self.index[i]
        p.apply(j, a, x_i, plan_next)

    def advance(self, dt):
        for p in self.plants:
            p.advance(dt)
