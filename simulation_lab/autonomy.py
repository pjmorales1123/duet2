"""Ground-truth lift/return teacher. Commands actuators only; never edits prop state.

IK uses a separate scratch MjData. Five arm coordinates satisfy position plus
alignment of a configurable gripper axis. The default local-Y-up mode preserves
the chemistry teacher; dinner grasps can also constrain the closing direction.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import mujoco
import numpy as np

from .scene import HOME, TABLE_Z
from .slots import blockers, tube_slot

GRASP_POINT = np.array([.003, 0., -.092])
OPEN, CLOSED = .4, -.08
GRIP_TORQUE = .15  # Nm; software torque saturation of the original position servo.
STAGES = ["planning", "approach", "descend", "close", "lift", "hold", "align", "lower", "release", "retract", "park", "verify"]
LABELS = {
    "idle": "Ready for a goal", "planning": "Check reach and clearance",
    "approach": "Move above the tube", "descend": "Align the open fingers",
    "close": "Close and verify both fingers", "retry": "Reopen for one grasp retry",
    "lift": "Lift clear of the rack", "hold": "Verify an unsupported hold",
    "rotate": "Orient the held object",
    "clearance": "Raise for transfer", "transit": "Carry to the destination rack",
    "align": "Center over the target slot", "lower": "Lower until the rack supports the tube",
    "release": "Open and clear the fixed finger", "retract": "Withdraw from the rack",
    "park": "Park the working arm", "verify": "Check stable placement after release",
}


def smooth(a):
    a = np.clip(a, 0., 1.)
    return a**3 * (10 + a * (-15 + 6*a))


def skew(v):
    x, y, z = v
    return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])


class PlanningError(ValueError):
    pass


class ArmIK:
    def __init__(self, model, data, side):
        self.model = model
        self.data = mujoco.MjData(model)
        self.data.qpos[:] = data.qpos
        self.offset = 0 if side == "left" else 6
        self.indices = np.arange(self.offset, self.offset + 5)
        self.body = model.body(f"{side}_gripper").id
        self.lo = model.jnt_range[self.indices, 0] + .004
        self.hi = model.jnt_range[self.indices, 1] - .004
        self.jp = np.zeros((3, model.nv))
        self.jr = self.jp.copy()
        self.grasp_point = GRASP_POINT.copy()
        self.x_target = None
        self.axis_index = 1
        self.axis_target = np.array([0., 0., 1.])
        self.axis_local = None

    def point(self, data):
        return data.xpos[self.body] + data.xmat[self.body].reshape(3, 3) @ self.grasp_point

    def solve(self, position, initial):
        q = np.asarray(initial).copy()
        for _ in range(180):
            self.data.qpos[self.indices] = q
            mujoco.mj_kinematics(self.model, self.data)
            mujoco.mj_comPos(self.model, self.data)
            point = self.point(self.data)
            rotation = self.data.xmat[self.body].reshape(3, 3)
            axis = rotation[:, self.axis_index] if self.axis_local is None else rotation @ self.axis_local
            error = np.r_[position - point, .08 * (self.axis_target - axis)]
            if self.x_target is not None:
                xaxis = self.data.xmat[self.body].reshape(3, 3)[:, 0]
                error = np.r_[error, .08*(self.x_target-xaxis)]
            if np.linalg.norm(error) < 1e-5:
                return q
            mujoco.mj_jac(self.model, self.data, self.jp, self.jr, point, self.body)
            jac = np.vstack((self.jp[:, self.indices], -.08 * skew(axis) @ self.jr[:, self.indices]))
            if self.x_target is not None:
                jac = np.vstack((jac, -.08*skew(xaxis) @ self.jr[:, self.indices]))
            delta = jac.T @ np.linalg.solve(jac @ jac.T + np.eye(len(error)) * 1e-5, error)
            q = np.clip(q + np.clip(delta, -.06, .06), self.lo, self.hi)
        raise PlanningError("The requested gripper pose is outside this arm's useful reach.")


@dataclass
class Motion:
    points: np.ndarray  # arm joint targets, in path order
    duration: float
    started: float
    grip_from: float
    grip_to: float

    def sample(self, now):
        fraction = float(smooth((now - self.started) / self.duration))
        n = fraction * (len(self.points) - 1)
        i = min(int(n), len(self.points) - 2)
        q = self.points[i] + (self.points[i + 1] - self.points[i]) * (n - i)
        return np.r_[q, self.grip_from + (self.grip_to - self.grip_from) * fraction]


class LiftReturn:
    def __init__(self, model, data, layout):
        self.model, self.data, self.layout = model, data, layout
        self.status = "idle"
        self.stage = "idle"
        self.message = "Load a practice scene and choose a robot goal."
        self.side = None
        self.tube = None
        self.started = self.stage_started = float(data.time)
        self.attempts = 0
        self.history = []
        self.metrics = {}
        self.request_pause = False
        self.motion = None
        self.last_check = -1.
        self.bad_grip_since = None
        self.stable_since = None
        self.max_lift = 0.
        self.result_written = False
        self.kind = "lift_return"
        self.stages = STAGES.copy()
        self.source_slot = self.destination = None
        self.recording_id = None

    @property
    def active(self):
        return self.status == "running"

    def snapshot(self):
        index = self.stages.index(self.stage) if self.stage in self.stages else 0
        return {"status": self.status, "active": self.active, "stage": self.stage,
                "stage_label": LABELS.get(self.stage, self.stage), "message": self.message,
                "arm": self.side, "tube_id": self.tube["id"] if self.tube else None,
                "kind": self.kind, "source_slot": self.source_slot["id"] if self.source_slot else None,
                "recording_id": self.recording_id,
                "destination_slot": self.destination["id"] if self.destination else None,
                "progress": 1. if self.status == "succeeded" else index / len(self.stages),
                "stages": [{"id": s, "label": LABELS[s]} for s in self.stages],
                "elapsed_s": round(float(self.data.time) - self.started, 2) if self.active else getattr(self, "elapsed", 0.),
                "attempts": self.attempts, "metrics": dict(self.metrics), "history": list(self.history),
                "observation": "exact_simulator_state", "attachment": "physical_contacts_only"}

    def _stage(self, stage, message=None):
        self.stage, self.stage_started = stage, float(self.data.time)
        self.message = message or LABELS[stage]
        self.stable_since = None
        self.history.append({"stage": stage, "at_s": round(self.data.time - self.started, 3), "message": self.message})

    def _finish(self, status, message, pause=False):
        self.status, self.message = status, message
        self.elapsed = round(float(self.data.time) - self.started, 3)
        self.request_pause = pause
        self.history.append({"stage": status, "at_s": self.elapsed, "message": message})

    def start(self, side="auto", tube_id=None, kind="lift_return", destination_slot=None):
        if self.active:
            raise ValueError("A goal is already running. Cancel it before starting another.")
        if tube_id and not any(t["id"] == tube_id for t in self.layout["tubes"]):
            raise ValueError("That tube does not exist in this scene.")
        if kind not in ("lift_return", "transfer"):
            raise ValueError("Unknown manipulation goal.")
        if destination_slot and not any(s["id"] == destination_slot for s in self.layout["slots"]):
            raise ValueError("That destination slot does not exist.")
        if kind == "lift_return" and destination_slot:
            raise ValueError("Lift & return uses the tube's current slot; select Transfer for another destination.")
        self.__init__(self.model, self.data, self.layout)
        self.requested_side, self.requested_tube = side, tube_id
        self.kind, self.requested_destination = kind, destination_slot
        if kind == "transfer":
            self.stages[6:6] = ["clearance", "transit"]
        self.status = "running"
        self._stage("planning")

    def cancel(self, targets):
        if self.active:
            targets[:] = self.data.qpos[:12]
            self._finish("cancelled", "Cancelled. Physics is paused to retain the current state. Reset the scene or resume with manual controls.", True)

    def _select(self, side, tube):
        self.side, self.tube = side, tube
        self.offset = 0 if side == "left" else 6
        self.ik = ArmIK(self.model, self.data, side)
        self.body = self.model.body(tube["body"]).id
        self.rack_body = self.model.body("rack_" + tube["rack"]).id
        self.base_geom = self.model.geom("rack_" + tube["rack"] + "_base").id
        self.moving_body = self.model.body(side + "_moving_jaw_so101_v1").id
        self.arm_geoms = set()
        self.fixed_geoms, self.moving_geoms = set(), set()
        for i in range(self.model.ngeom):
            body = int(self.model.geom_bodyid[i])
            if self.model.body(body).name.startswith(side + "_"):
                self.arm_geoms.add(i)
            name = self.model.geom(i).name or ""
            if name.startswith(side + "_fixed_jaw") or (body == self.ik.body and self.model.geom_group[i] == 4):
                self.fixed_geoms.add(i)
            if body == self.moving_body:
                self.moving_geoms.add(i)
        self.jaw_geoms = self.fixed_geoms | self.moving_geoms
        self.origin = self.data.xpos[self.body].copy()
        self.source_slot = tube_slot(self.data, self.layout, tube)
        self.destination = self.source_slot
        self.destination_position = self.origin.copy()
        self.others = {t["id"]: self.data.body(t["body"]).xpos.copy() for t in self.layout["tubes"] if t != tube}
        self.parked = self.data.qpos[6:12].copy() if side == "left" else self.data.qpos[:6].copy()

    def _collision(self, data, allow_tube=False, penetration=.00015):
        for c in data.contact:
            a, b = int(c.geom1), int(c.geom2)
            if c.dist >= -penetration or not ({a, b} & self.arm_geoms):
                continue
            if allow_tube and ((a in self.jaw_geoms and self.model.geom_bodyid[b] == self.body) or
                               (b in self.jaw_geoms and self.model.geom_bodyid[a] == self.body)):
                continue
            return f"{self.model.geom(a).name or self.model.body(self.model.geom_bodyid[a]).name} / {self.model.geom(b).name or self.model.body(self.model.geom_bodyid[b]).name}"
        return None

    def _carry_reference(self, q=None):
        if q is None:
            rotation = self.data.xmat[self.ik.body].reshape(3, 3)
            point = self.ik.point(self.data)
        else:
            self.ik.data.qpos[self.offset:self.offset+5] = q
            mujoco.mj_kinematics(self.model, self.ik.data)
            rotation = self.ik.data.xmat[self.ik.body].reshape(3, 3)
            point = self.ik.point(self.ik.data)
        return (rotation.T @ (self.data.xpos[self.body]-point),
                rotation.T @ self.data.xmat[self.body].reshape(3, 3))

    def _point_for_center(self, center, reference, initial):
        q = initial.copy()
        for _ in range(5):
            self.ik.data.qpos[self.offset:self.offset+5] = q
            mujoco.mj_kinematics(self.model, self.ik.data)
            rotation = self.ik.data.xmat[self.ik.body].reshape(3, 3)
            point = np.asarray(center)-rotation @ reference[0]
            q = self.ik.solve(point, q)
        return point, q

    def _check_path(self, points, grip, allow_tube=False, carry=None, support=None):
        scratch = self.ik.data
        scratch.qpos[:] = self.data.qpos
        scratch.qvel[:] = 0
        scratch.ctrl[:] = self.data.ctrl
        for q in points:
            scratch.qpos[self.offset:self.offset+5] = q
            scratch.qpos[self.offset+5] = grip
            if carry is not None:
                # Move the hypothetical carried object in scratch data only.
                mujoco.mj_kinematics(self.model, scratch)
                rotation = scratch.xmat[self.ik.body].reshape(3, 3)
                address = self.model.joint(self.tube["body"]+"_free").qposadr[0]
                scratch.qpos[address:address+3] = self.ik.point(scratch)+rotation @ carry[0]
                quat = np.empty(4)
                mujoco.mju_mat2Quat(quat, (rotation @ carry[1]).ravel())
                scratch.qpos[address+3:address+7] = quat
            mujoco.mj_forward(self.model, scratch)
            collision = self._collision(scratch, allow_tube)
            if collision:
                raise PlanningError("The planned path has insufficient clearance: " + collision)
            if carry is not None:
                for c in scratch.contact:
                    a, b = int(c.geom1), int(c.geom2)
                    if self.model.geom_bodyid[a] != self.body and self.model.geom_bodyid[b] != self.body:
                        continue
                    other = b if self.model.geom_bodyid[a] == self.body else a
                    if c.dist < -.00015 and other not in self.jaw_geoms and other != support:
                        raise PlanningError("The carried tube's path intersects " + (self.model.geom(other).name or "another object") + ".")

    def _cartesian(self, start, end, initial, grip, check=True):
        count = max(3, int(np.linalg.norm(end - start) / .002) + 2)
        points, q = [], initial.copy()
        for a in np.linspace(0., 1., count):
            q = self.ik.solve(start + (end - start) * a, q)
            points.append(q.copy())
        points = np.array(points)
        if check:
            self._check_path(points, grip, allow_tube=True)
        return points

    def _joint_path(self, start, end, grip):
        points = np.linspace(start, end, max(8, int(np.max(np.abs(end-start))/.035)+2))
        self._check_path(points, grip)
        return points

    def _move(self, stage, points, grip, duration):
        # Quintic time scaling, with conservative limits for both joint and hand travel.
        gradient = np.abs(np.diff(points, axis=0)).max() * (len(points)-1)
        duration = max(duration, float(1.875 * gradient / .8))
        self.motion = Motion(points, duration, float(self.data.time), float(self.data.qpos[self.offset+5]), grip)
        self._stage(stage)

    def _move_point(self, stage, end, grip, duration):
        points = self._cartesian(self.ik.point(self.data), np.asarray(end), self.data.qpos[self.offset:self.offset+5], grip, check=False)
        carry = self._carry_reference() if self.kind == "transfer" and stage in ("clearance", "transit", "align", "lower") else None
        self._check_path(points, grip, allow_tube=True, carry=carry, support=self.base_geom if stage == "lower" else None)
        self._move(stage, points, grip, duration)

    def _choose_destination(self, q, above):
        if self.kind == "lift_return":
            self.base_geom = self.model.geom("rack_"+self.source_slot["rack"]+"_base").id
            return
        options = [s for s in self.layout["slots"] if not self.requested_destination or s["id"] == self.requested_destination]
        options.sort(key=lambda s: (s["index"] != 1, np.linalg.norm(np.array(s["position_m"][:2])-self.origin[:2])))
        failures = []
        reference = self._carry_reference(q)
        for slot in options:
            try:
                if slot["rack"] == self.source_slot["rack"]:
                    raise PlanningError("A transfer destination must be in a different rack.")
                occupied = blockers(self.data, self.layout, slot, self.tube["id"])
                if occupied:
                    raise PlanningError(f"Destination {slot['id']} is occupied or blocked by {', '.join(occupied)}.")
                base = self.model.geom("rack_"+slot["rack"]+"_base").id
                dest = np.array([*slot["position_m"][:2], self.origin[2]+slot["position_m"][2]-self.source_slot["position_m"][2]])
                carry_point = self.grasp+[0, 0, .080]
                qc = self.ik.solve(carry_point, above)
                transit_point, qt = self._point_for_center(dest+[0, 0, .080], reference, qc)
                transit = self._cartesian(carry_point, transit_point, qc, CLOSED, check=False)
                self._check_path(transit, CLOSED, allow_tube=True, carry=reference)
                place_point, _ = self._point_for_center(dest, reference, qt)
                descent = self._cartesian(transit_point, place_point, qt, CLOSED, check=False)
                self._check_path(descent, CLOSED, allow_tube=True, carry=reference, support=base)
                self.destination, self.destination_position, self.base_geom = slot, dest, base
                return
            except PlanningError as exc:
                failures.append(str(exc))
        raise PlanningError("No usable destination. " + (failures[0] if failures else "No destination slots."))

    def _destination_is_clear(self):
        if self.kind == "transfer":
            occupied = blockers(self.data, self.layout, self.destination, self.tube["id"])
            if occupied:
                raise PlanningError("The reserved destination is now blocked by " + ", ".join(occupied) + ".")

    def _plan(self, targets):
        if np.max(np.abs(self.data.qpos[:12] - np.array(HOME*2))) > .10 or np.max(np.abs(self.data.qvel[:12])) > .12:
            raise PlanningError("Start with both arms parked. Use Home both arms, wait for them to settle, or reset the scene.")
        candidates = []
        for tube in self.layout["tubes"]:
            if self.requested_tube and tube["id"] != self.requested_tube:
                continue
            for side in ("left", "right") if self.requested_side == "auto" else (self.requested_side,):
                base = np.array([-.25 if side == "left" else .25, -.235])
                distance = np.linalg.norm(self.data.body(tube["body"]).xpos[:2] - base)
                candidates.append((distance, side, tube))
        reasons = []
        for _, side, tube in sorted(candidates, key=lambda c: (c[0], c[1], c[2]["id"])):
            self._select(side, tube)
            try:
                if not self.source_slot or self.data.xmat[self.body, 8] < math.cos(math.radians(7)) or np.linalg.norm(self.origin[:2] - np.array(self.source_slot["position_m"][:2])) > .005:
                    raise PlanningError("The tube is tilted or displaced; reset its rack before this first skill.")
                p = self.origin.copy()
                p[2] += tube["height"]/2 - .008
                self.grasp = p
                self.hover = p + [0., 0., .060]
                q = self.ik.solve(p, HOME[:5])
                above = self.ik.solve(self.hover, q)
                approach = self._joint_path(self.data.qpos[self.offset:self.offset+5], above, OPEN)
                descent = self._cartesian(self.hover, p, above, OPEN, check=False)
                self._check_path(descent, OPEN, allow_tube=False)
                self._check_path(np.array([q]), OPEN, allow_tube=False)
                self._choose_destination(q, above)
                self.metrics = {"max_lift_cm": 0., "hold_verified_s": 0., "both_fingers": False,
                                "other_arm_max_motion_deg": 0., "unexpected_collisions": 0,
                                "gripper_torque_limit_nm": GRIP_TORQUE, "ik_axis": "gripper_y_up"}
                targets[:] = HOME*2
                self.attempts = 1
                self._move("approach", approach, OPEN, 3.)
                self.message = f"{side.title()} arm: tube {tube['id']} from {self.source_slot['id']} to {self.destination['id']}. Reach and clearance checked."
                return
            except PlanningError as exc:
                reasons.append(f"{side} {tube['id']}: {exc}")
        self.side = None
        self.tube = None
        self.destination = self.source_slot = None
        raise PlanningError("No clear reachable grasp. Load the lift practice scene or move the racks. " + (reasons[0] if reasons else "No tube candidates."))

    def _observe(self):
        forces = {"fixed": 0., "moving": 0., "base": 0., "external": 0.}
        wrench = np.zeros(6)
        for index, c in enumerate(self.data.contact):
            a, b = int(c.geom1), int(c.geom2)
            if self.model.geom_bodyid[a] != self.body and self.model.geom_bodyid[b] != self.body:
                continue
            other = b if self.model.geom_bodyid[a] == self.body else a
            mujoco.mj_contactForce(self.model, self.data, index, wrench)
            force = max(0., float(wrench[0]))
            if other in self.fixed_geoms:
                forces["fixed"] += force
            elif other in self.moving_geoms:
                forces["moving"] += force
            else:
                forces["external"] += force
            if other == self.base_geom:
                forces["base"] += force
        pos = self.data.xpos[self.body]
        lift = float(pos[2] - self.origin[2])
        up = float(self.data.xmat[self.body, 8])
        extent = (self.tube["height"]/2*abs(up) + .009*math.sqrt(max(0., 1-up*up))) if self.layout["practice"] else (.009 + (self.tube["height"]/2-.009)*abs(up))
        bottom = float(pos[2]-extent)
        both = min(forces["fixed"], forces["moving"]) > .08
        self.max_lift = max(self.max_lift, lift)
        self.metrics.update(lift_cm=round(lift*100, 3), max_lift_cm=round(self.max_lift*100, 3),
                            both_fingers=both, finger_forces_n=[round(forces["fixed"], 3), round(forces["moving"], 3)],
                            rack_support_n=round(forces["base"], 3), external_contact_n=round(forces["external"], 3),
                            tilt_deg=round(math.degrees(math.acos(np.clip(up, -1, 1))), 3))
        parked = self.data.qpos[6:12] if self.side == "left" else self.data.qpos[:6]
        self.metrics["other_arm_max_motion_deg"] = round(max(self.metrics["other_arm_max_motion_deg"], float(np.rad2deg(np.max(np.abs(parked-self.parked))))), 4)
        return forces, lift, up, bottom, both

    def _done_motion(self):
        return self.data.time >= self.motion.started + self.motion.duration + .3 and np.max(np.abs(self.data.qpos[self.offset:self.offset+5] - self.motion.points[-1])) < .015

    def apply_gripper_limit(self, targets):
        """Equivalent torque saturation without changing the shared MjModel."""
        ctrl = targets.copy()
        if self.active and self.side:
            i = self.offset + 5
            kp = self.model.actuator_gainprm[i, 0]
            kv = -self.model.actuator_biasprm[i, 2]
            torque = kp*(targets[i]-self.data.qpos[i]) - kv*self.data.qvel[i]
            ctrl[i] = self.data.qpos[i] + (np.clip(torque, -getattr(self, "grip_torque", GRIP_TORQUE), getattr(self, "grip_torque", GRIP_TORQUE)) + kv*self.data.qvel[i])/kp
        return ctrl

    def update(self, targets):
        if not self.active:
            return
        try:
            if self.stage == "planning":
                self._plan(targets)
                return
            now = float(self.data.time)
            if now-self.started > 75.:
                raise PlanningError("The goal exceeded its 75-second simulation-time limit.")
            if now-self.stage_started > (self.motion.duration+4. if self.motion else 8.):
                raise PlanningError("Timed out during " + LABELS[self.stage].lower() + ".")
            targets[self.offset:self.offset+6] = self.motion.sample(now)
            # Contact and placement checks use every fixed physics tick, not camera frames.
            forces, lift, up, bottom, both = self._observe()
            collision = self._collision(self.data, allow_tube=True, penetration=.0008)
            if collision:
                self.metrics["unexpected_collisions"] += 1
                raise PlanningError("Unexpected contact; stopped before continuing: " + collision)
            if self.metrics["other_arm_max_motion_deg"] > 1.:
                raise PlanningError("The parked arm moved unexpectedly.")
            if self.stage in ("lift", "hold", "clearance", "transit", "align", "lower") and lift > .012:
                if not both:
                    self.bad_grip_since = self.bad_grip_since or now
                    if now-self.bad_grip_since > .18:
                        raise PlanningError("Lost contact at one or both fingers. The tube is no longer verified as grasped.")
                else:
                    self.bad_grip_since = None
                if up < math.cos(math.radians(15)):
                    raise PlanningError("The held tube tilted beyond the 15-degree retention limit.")
            if self.stage in ("clearance", "transit", "align") and self.kind == "transfer" and forces["external"] > .10:
                raise PlanningError("The carried tube contacted another object during transfer.")
            if self.stage == "approach" and self._done_motion():
                self._move_point("descend", self.grasp, OPEN, 3.)
            elif self.stage == "descend" and self._done_motion():
                q = self.data.qpos[self.offset:self.offset+5].copy()
                self._move("close", np.array([q, q]), CLOSED, 5.)
            elif self.stage == "close" and self._done_motion():
                if both and up > math.cos(math.radians(7)):
                    self._move_point("lift", self.ik.point(self.data)+[0., 0., .060], CLOSED, 3.5)
                elif self.attempts < 2 and np.linalg.norm(self.data.xpos[self.body][:2]-self.origin[:2]) < .006:
                    self.attempts += 1
                    q = self.data.qpos[self.offset:self.offset+5].copy()
                    self._move("retry", np.array([q, q]), OPEN, 4.)
                    self.message = "Grasp not verified. Reopening once before a bounded retry."
                else:
                    raise PlanningError("Both fingers did not establish a stable grasp after the allowed attempts.")
            elif self.stage == "retry" and self._done_motion():
                self.grasp = self.data.xpos[self.body].copy()+[0, 0, self.tube["height"]/2-.008]
                self._move_point("descend", self.grasp, OPEN, 1.5)
            elif self.stage == "lift" and self._done_motion():
                if not (both and .050 <= lift <= .075 and bottom > TABLE_Z+.061 and forces["external"] < .02):
                    raise PlanningError("Lift was not independently supported by both fingers above the rack.")
                q = self.data.qpos[self.offset:self.offset+5].copy()
                self._move("hold", np.array([q, q]), CLOSED, 1.5)
                self.hold_start_height = lift
            elif self.stage == "hold":
                if not both or forces["external"] >= .02 or bottom <= TABLE_Z+.061 or lift < .050 or self.hold_start_height-lift > .004:
                    raise PlanningError("The tube slipped or regained external support during the hold.")
                self.metrics["hold_verified_s"] = round(now-self.stage_started, 3)
                if now-self.stage_started >= 1.5:
                    self.metrics["hold_min_lift_cm"] = round(lift*100, 3)
                    self._destination_is_clear()
                    if self.kind == "transfer":
                        self._move_point("clearance", self.ik.point(self.data)+[0, 0, .025], CLOSED, 2.)
                        return
                    p = self.ik.point(self.data).copy()
                    p[:2] += self.destination_position[:2]-self.data.xpos[self.body,:2]
                    self._move_point("align", p, CLOSED, 1.5)
            elif self.stage == "clearance" and self._done_motion():
                self._destination_is_clear()
                center = self.data.xpos[self.body].copy()
                center[:2] = self.destination_position[:2]
                point, _ = self._point_for_center(center, self._carry_reference(), self.data.qpos[self.offset:self.offset+5])
                duration = max(4., 1.875*np.linalg.norm(point-self.ik.point(self.data))/.08)
                self._move_point("transit", point, CLOSED, duration)
            elif self.stage == "transit" and self._done_motion():
                self._destination_is_clear()
                center = self.data.xpos[self.body].copy()
                center[:2] = self.destination_position[:2]
                point, _ = self._point_for_center(center, self._carry_reference(), self.data.qpos[self.offset:self.offset+5])
                self._move_point("align", point, CLOSED, 1.5)
            elif self.stage == "align" and self._done_motion():
                self._destination_is_clear()
                p = self.ik.point(self.data).copy()
                p[2] += self.destination_position[2]-.0005-self.data.xpos[self.body, 2]
                self._move_point("lower", p, CLOSED, 4.)
            elif self.stage == "lower":
                # Stop descent at measured base support, avoiding compression of the tube.
                if self.data.xpos[self.body, 2]-self.destination_position[2] < .003 and forces["base"] > .06:
                    p = self.ik.point(self.data) - self.data.xmat[self.ik.body].reshape(3, 3)[:, 0]*.010
                    self._move_point("release", p, OPEN, 4.)
                elif self._done_motion():
                    raise PlanningError("The tube did not settle onto the rack base; refusing to release it.")
            elif self.stage == "release" and self._done_motion():
                if self.data.qpos[self.offset+5] < .35:
                    return
                p = self.ik.point(self.data) + [0., 0., .065]
                self._move_point("retract", p, OPEN, 3.)
            elif self.stage == "retract" and self._done_motion():
                points = self._joint_path(self.data.qpos[self.offset:self.offset+5], np.array(HOME[:5]), OPEN)
                self._move("park", points, HOME[5], 3.)
            elif self.stage == "park" and self._done_motion():
                q = self.data.qpos[self.offset:self.offset+5].copy()
                self._move("verify", np.array([q, q]), HOME[5], 1.)
            elif self.stage == "verify":
                self._verify_placement(forces, up, now)
        except (PlanningError, np.linalg.LinAlgError) as exc:
            targets[:] = self.data.qpos[:12]
            self._finish("failed", str(exc) + " Physics paused; reset or inspect before retrying.", True)

    def _verify_placement(self, forces, up, now):
        position = self.data.xpos[self.body]
        xy = float(np.linalg.norm(position[:2] - np.array(self.destination["position_m"][:2])))
        z = abs(float(position[2]-self.destination_position[2]))
        j = self.model.joint(self.tube["body"]+"_free").dofadr[0]
        speed = float(np.linalg.norm(self.data.qvel[j:j+3]))
        angular = float(np.linalg.norm(self.data.qvel[j+3:j+6]))
        disturbed = max((float(np.linalg.norm(self.data.body(t["body"]).xpos-self.others[t["id"]])) for t in self.layout["tubes"] if t != self.tube), default=0.)
        valid = xy < .0045 and z < .002 and up > math.cos(math.radians(10)) and speed < .003 and angular < .08 and forces["base"] > .08 and forces["fixed"]+forces["moving"] < .01 and disturbed < .003
        self.metrics.update(placement_xy_error_mm=round(xy*1000, 3), placement_z_error_mm=round(z*1000, 3),
                            placement_speed_mm_s=round(speed*1000, 3), other_tube_max_displacement_mm=round(disturbed*1000, 3))
        if valid:
            if self.stable_since is None:
                self.stable_since = now
            if now-self.stable_since >= 1.:
                self.metrics["placement_stable_s"] = round(now-self.stable_since, 3)
                where = f"transferred upright from {self.source_slot['id']} to {self.destination['id']}" if self.kind == "transfer" else "returned upright to its original slot"
                self._finish("succeeded", f"Tube {self.tube['id']} lifted, held clear for 1.5 s, and {where}. Both arms are parked.")
        else:
            self.stable_since = None
        if now-self.stage_started > 4.:
            raise PlanningError("Placement was not stable, upright and centered after the fingers withdrew.")
