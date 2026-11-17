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

    def apply(self, i, a, x_i=None):
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
        self.alt, self.log, self.timeout, self.dt = alt, log, timeout, dt
        self.x = np.zeros((len(urls), 4)); self.t_boot = np.zeros(len(urls))
        self.att = np.zeros((len(urls), 3)); self.act = np.full((len(urls), 4), np.nan)
        for L in self.links:
            if L.wait_heartbeat(timeout=60) is None:
                raise RuntimeError("no heartbeat on " + str(L))
        self.cmd = np.zeros((len(urls), 4))      # vx, vy, ax, ay
        self._poll(0.5)

    # --- telemetry ---
    def _poll(self, dur=0.0):
        end = time.time() + dur
        while True:
            for k, L in enumerate(self.links):
                while (m := L.recv_match(blocking=False)) is not None:
                    t = m.get_type()
                    if t == "LOCAL_POSITION_NED":
                        self.x[k] = [m.x + self.home[k, 0], m.y + self.home[k, 1], m.vx, m.vy]
                        self.t_boot[k] = m.time_boot_ms / 1e3
                        self.z = getattr(self, "z", np.zeros(len(self.links))); self.z[k] = -m.z
                    elif t == "ATTITUDE":
                        self.att[k] = [m.roll, m.pitch, m.yaw]
                    elif t == "HIL_ACTUATOR_CONTROLS":
                        self.act[k] = m.controls[:4]
            if time.time() >= end:
                return
            time.sleep(0.002)

    def states(self):
        return self.x.copy()

    # --- commands ---
    def _setpoint(self, k):
        L = self.links[k]
        vx, vy, ax, ay = self.cmd[k]
        # use vx, vy, z position and ax, ay feed-forward; ignore x, y position and yaw rate
        mask = (1 << 0) | (1 << 1) | (1 << 5) | (1 << 8) | (1 << 10) | (1 << 11)
        L.mav.set_position_target_local_ned_send(0, L.target_system, L.target_component,
                                                 self.mavutil.mavlink.MAV_FRAME_LOCAL_NED, mask,
                                                 0, 0, -self.alt, vx, vy, 0, ax, ay, 0, 0, 0)

    def apply(self, i, a, x_i):
        v = np.asarray(x_i[2:]) + np.asarray(a) * self.dt
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
    def arm_takeoff(self, climb_s=25.0):
        mav = self.mavutil.mavlink
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
            if np.all(getattr(self, "z", np.zeros(1)) > self.alt - 2):
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

    def apply(self, i, a, x_i):
        p, j = self.index[i]
        p.apply(j, a, x_i)

    def advance(self, dt):
        for p in self.plants:
            p.advance(dt)
