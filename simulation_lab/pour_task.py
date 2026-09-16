"""Two-arm 'put water' skill with a table-supported bottle relay.

The bottle is relayed right-to-left, the right arm grips and holds the cup
completely still, and only the left bottle arm approaches and tilts. Both
objects return to their table settings without object-state writes.
"""
from copy import deepcopy
import math
import numpy as np
from .autonomy import PlanningError
from .dinner_autonomy import CLOSED, DinnerSequence, DinnerTask

BOTTLE_RELAY_POINT = (0., -.14)
MUG_HOLD_XY = (.16, -.08)
MUG_POUR_XY = (.10, -.08)
MUG_HOLD_HEIGHT = .05
POUR_OFFSET = np.array([0., -.085, .12])
POUR_STANDOFF = .085
POUR_DESCENT = .10
POUR_TILT_RAD = .9  # ~50 degrees, tips the neck toward the cup

def _with_bottle_at_relay_point(layout):
    layout = deepcopy(layout)
    target = next(t for t in layout['targets'] if t['object_id'] == 'bottle')
    target['position_m'] = [BOTTLE_RELAY_POINT[0], BOTTLE_RELAY_POINT[1], layout['table_z']]
    return layout


class BottleDonorTask(DinnerTask):
    """Right arm places the bottle at the shared table relay point."""

    def _move_center(self, stage, center, grip, duration):
        if stage == 'align':
            center = center.copy()
            center[2] += .025
        super()._move_center(stage, center, grip, duration)


class HoldTask(DinnerTask):
    """Picks the object and carries it to hold_point (or, if unset, just
    above where it already is), staying airborne there - never resting on
    the table - until told (via cleared_to_place) to finish the normal
    place cycle back at the object's real table setting."""
    hold_point = None  # optional (x, y, z) override

    def _move_center(self, stage, center, grip, duration):
        if stage == 'align' and self.hold_point is not None:
            center = np.asarray(self.hold_point)
        if stage == 'lower' and not getattr(self, 'cleared_to_place', False):
            return  # keep holding; the last commanded target already holds steady
        if stage == 'lower' and self.hold_point is not None:
            current = self.data.xpos[self.body].copy()
            waypoints = [
                np.array([center[0], center[1], current[2]]),
                np.asarray(center),
            ]
            reference = self._carry_reference()
            current_point = self.ik.point(self.data).copy()
            q = self.data.qpos[self.offset:self.offset+5].copy()
            paths = []
            for waypoint in waypoints:
                point, _ = self._point_for_center(waypoint, reference, q)
                path = self._cartesian(current_point, point, q, grip, check=False)
                paths.append(path)
                current_point, q = point, path[-1]
            points = np.vstack(paths)
            self._check_path(points, float(self.data.qpos[self.offset+5]), True, carry=reference, support=self.base_geom)
            self._move(stage, points, grip, duration)
            return
        super()._move_center(stage, center, grip, duration)


class PourBottleTask(DinnerTask):
    """Picks the bottle from the relay point and lifts it, then PARKS nearby
    (small retreat, no ambitious reach) and waits - the mug hasn't moved into
    the shared pour station yet, and it's much too close to the relay point
    for both events to happen at once without the arms colliding. Once told
    (via cleared_to_pour) that the mug is in place, it routes laterally
    around the stationary cup wrist, tilts and pours, then returns to its own
    setting - same as before, just no longer racing the mug for the same
    patch of table."""
    cleared_to_pour = False

    def __init__(self, model, data, layout, mug_body_id=None):
        super().__init__(model, data, layout)
        self.mug_body_id = mug_body_id

    def _select_item(self, side, item):
        super()._select_item(side, item)
        self.grip_torque = .65
        self.metrics['gripper_torque_limit_nm'] = self.grip_torque

    def start(self, **kwargs):
        mug_body_id = self.mug_body_id
        super().start(**kwargs)
        self.mug_body_id = mug_body_id
        self.cleared_to_pour = False

    def _move_center(self, stage, center, grip, duration):
        if stage == 'align' and getattr(self, 'poured', False):
            center = center.copy()
            center[2] += .025  # clears the plate's rim on the way back to the relay point
            super()._move_center(stage, center, grip, duration)
            return
        if stage == 'align' and not getattr(self, 'poured', False) and not self.cleared_to_pour:
            self._final_align = (center, grip, duration)
            park = self.data.xpos[self.body].copy()
            park[2] += .075  # retreat with real table clearance - just holds here until the mug is out of the way
            super()._move_center(stage, park, grip, duration)
            return
        if stage == 'lower' and not self.cleared_to_pour:
            return  # keep parked; the last commanded target already holds steady
        if stage == 'align' and not getattr(self, 'poured', False) and self.cleared_to_pour:
            self._begin_pour_approach(grip)
            return
        super()._move_center(stage, center, grip, duration)

    def _begin_pour_approach(self, grip):
        pour_anchor = np.array([MUG_POUR_XY[0], MUG_POUR_XY[1], self.data.xpos[self.mug_body_id][2]])
        self.pour_target = pour_anchor + POUR_OFFSET
        start = self.ik.point(self.data).copy()
        # Traverse high, then descend at the cup.  The prior version never
        # performed this descent: it tilted with the mouth far above and
        # behind the cup, which looked like a pour but could not pour into it.
        pour_point = self.pour_target.copy()
        # Descend at the clear stand-off point before tilting.  The mug is
        # deliberately staged out of the neck's sweep until tilt completes.
        pour_point[2] = start[2] - POUR_DESCENT
        self.pour_target[2] = start[2] - POUR_DESCENT
        offset_direction = POUR_OFFSET[:2] / np.linalg.norm(POUR_OFFSET[:2])
        pour_point[:2] = pour_anchor[:2] + offset_direction * POUR_STANDOFF
        # Escape away from the stationary cup wrist before traversing, rather
        # than cutting the diagonal corner through its camera housing.
        bottle_radius = self.tube['size_m'][0]/2
        camera_gid = self.model.geom('right_camera_box2').id
        camera_center = self.data.geom_xpos[camera_gid].copy()
        camera_half_y = self.model.geom_size[camera_gid][1]
        bypass_y = min(start[1], camera_center[1] - camera_half_y - bottle_radius - .015)
        escape = np.array([start[0], bypass_y])
        traverse = np.array([pour_point[0], bypass_y])
        # The bottle stays high above the plate, so validate its real 3-D
        # geometry below instead of rejecting the final cup approach with a
        # deliberately conservative 2-D plate footprint.
        route = [start[:2], escape, traverse, pour_point[:2]]
        reference = self._carry_reference()
        current_point = start
        q = self.data.qpos[self.offset:self.offset+5].copy()
        paths = []
        for xy in route[1:]:
            target = np.array([xy[0], xy[1], start[2]])
            path = self._cartesian(current_point, target, q, grip, check=False)
            paths.append(path)
            current_point, q = target, path[-1]
        descent = self._cartesian(current_point, pour_point, q, grip, check=False)
        paths.append(descent)
        points = np.vstack(paths)
        self._check_path(points, grip, True, carry=reference)
        self._move('pour_approach', points, grip, 4.)

    def update(self, targets):
        if self.stage in ('pour_approach', 'pour_tilt', 'await_cup', 'pour_hold', 'pour_return') and self.active:
            self._pour_update(targets)
            return
        super().update(targets)

    def _pour_update(self, targets):
        try:
            now = float(self.data.time)
            if now-self.started > 90 or now-self.stage_started > self.motion.duration+5:
                raise PlanningError('Motion timed out: '+self.stage)
            targets[self.offset:self.offset+6] = self.motion.sample(now)
            forces, lift, up, both = self._observe()
            collision = self._collision(self.data, True, .0008)
            if collision:
                self.metrics['unexpected_collisions'] += 1
                raise PlanningError('Unexpected contact: '+collision)
            if not both:
                self.bad_grip_since = self.bad_grip_since or now
                if now-self.bad_grip_since > .18:
                    raise PlanningError('Lost verified finger contact while pouring.')
            else:
                self.bad_grip_since = None
            done = self._done_motion()
            q = self.data.qpos[self.offset:self.offset+5].copy()
            point = self.ik.point(self.data).copy()
            if self.stage == 'pour_approach' and done:
                toward_cup = np.asarray(MUG_POUR_XY) - point[:2]
                toward_cup /= np.linalg.norm(toward_cup)
                self.ik.axis_target = np.array([
                    math.sin(POUR_TILT_RAD) * toward_cup[0],
                    math.sin(POUR_TILT_RAD) * toward_cup[1],
                    math.cos(POUR_TILT_RAD),
                ])
                end = self.ik.solve(point, q)
                path = np.linspace(q, end, 60)
                self._check_path(path, CLOSED, True, carry=self._carry_reference())
                self._move('pour_tilt', path, CLOSED, 2.5)
            elif self.stage == 'pour_tilt' and done:
                self._move('await_cup', np.array([q, q]), CLOSED, 30.)
            elif self.stage == 'pour_hold' and done:
                self._move('await_cup_clear', np.array([q, q]), CLOSED, 30.)
            elif self.stage == 'pour_return' and done:
                self.poured = True
                center, grip, duration = self._final_align
                self._move_center('align', center, grip, duration)
        except (PlanningError, np.linalg.LinAlgError) as exc:
            targets[:] = self.data.qpos[:12]
            self._finish('failed', str(exc)+' Physics paused for inspection.', True)

    def begin_pour_return(self):
        q = self.data.qpos[self.offset:self.offset+5].copy()
        point = self.ik.point(self.data).copy()
        self.ik.axis_target = np.array([0., 0., 1.])
        end = self.ik.solve(point, q)
        path = np.linspace(q, end, 60)
        self._check_path(path, CLOSED, True, carry=self._carry_reference())
        self._move('pour_return', path, CLOSED, 2.5)


class PourWaterTask(DinnerSequence):
    """Relay bottle right-to-left, hold cup still, pour left-to-right, return."""

    HOLD_VERIFIED_S = 1.6

    def start(self, **kwargs):
        if self.active:
            raise ValueError('Cancel the current goal first.')
        if self.layout['dinner_preset'] != 'task':
            raise ValueError('Load Task start before running a dinner goal.')
        layout = _with_bottle_at_relay_point(self.layout)
        self.__init__(self.model, self.data, layout)
        self.kind = 'pour_water'
        self.status = 'running'
        self._start_bottle_donor()

    def _start_bottle_donor(self):
        self.phase = 'bottle_relay'
        self.child = BottleDonorTask(self.model, self.data, self.layout)
        self.child.start(side='right', object_id='bottle')
        self.stage = self.child.stage

    def _start_left_lift_bottle(self):
        self.phase = 'left_lift_bottle'
        self.child = PourBottleTask(self.model, self.data, self.layout)
        self.child.start(side='left', object_id='bottle')
        # Hold the bottle around its full-width body.  This is tighter than
        # the former neck grasp and leaves enough bottle above the fingers
        # for the mouth to reach the cup without colliding the two wrists.
        self.child.grasp_local_override = np.array([0., 0., .105])
        self.stage = self.child.stage

    def _start_hold_mug(self):
        self.phase = 'hold_mug'
        self.bottle_child = self.child
        self.child = HoldTask(self.model, self.data, self.layout)
        self.child.hold_point = np.array([MUG_HOLD_XY[0], MUG_HOLD_XY[1], self.layout['table_z']+MUG_HOLD_HEIGHT])
        self.child.start(side='right', object_id='mug')
        self.child.preferred_grasp_directions = [np.array([1., 0.])]
        self.child.require_home_start = False
        self.child.other_arm_tolerance_deg = 180.
        self.stage = self.child.stage

    def _bottle_parked(self):
        return self.child.stage == 'align' and self.child.motion is not None and self.child._done_motion()

    def _start_pour(self):
        self.mug_child = self.child
        self.mug_child.others.pop('bottle', None)
        self.phase = 'pour'
        self.child = self.bottle_child
        self.child.mug_body_id = self.mug_child.body
        self.child.others.pop('mug', None)
        self.child.cleared_to_pour = True
        self.child.other_arm_tolerance_deg = 180.  # the right arm is deliberately holding the mug, not parked
        try:
            self.child._begin_pour_approach(CLOSED)
        except (PlanningError, ValueError) as exc:
            self.child._finish('failed', str(exc)+' Physics paused for inspection.', True)
        self.stage = self.child.stage

    def _start_present_mug(self):
        """Move the already-grasped mug under the measured bottle outlet."""
        self.bottle_child = self.child
        bottle_rotation = self.data.xmat[self.bottle_child.body].reshape(3, 3)
        mouth = self.data.xpos[self.bottle_child.body] + bottle_rotation @ np.array([0., 0., .16])
        # Keep the outlet inside the opening while biasing the cup 14 mm
        # toward the right arm, which keeps its wrist clear of the neck.
        self.mug_child.hold_point = np.array([mouth[0] + .014, mouth[1], self.layout['table_z'] + MUG_HOLD_HEIGHT])
        self.mug_child._move_center('align', self.mug_child.hold_point, CLOSED, 2.)
        self.child = self.mug_child
        self.phase = 'present_mug'
        self.stage = 'pour_insert'

    def _start_pour_hold(self):
        self.child = self.bottle_child
        self.phase = 'pour'
        q = self.data.qpos[self.child.offset:self.child.offset+5].copy()
        self.child._move('pour_hold', np.array([q, q]), CLOSED, 1.5)
        self.stage = self.child.stage

    def _start_retract_mug(self):
        self.bottle_child = self.child
        self.mug_child.hold_point = np.array([MUG_HOLD_XY[0], MUG_HOLD_XY[1], self.layout['table_z'] + MUG_HOLD_HEIGHT])
        self.mug_child._move_center('align', self.mug_child.hold_point, CLOSED, 2.)
        self.child = self.mug_child
        self.phase = 'retract_mug'
        self.stage = 'cup_clear'

    def _start_pour_return(self):
        self.child = self.bottle_child
        self.phase = 'pour'
        self.child.begin_pour_return()
        self.stage = self.child.stage

    def _mug_settled_at_hold_point(self):
        return self.child.stage == 'align' and self.child.motion is not None and self.child._done_motion()

    def update(self, targets):
        if not self.active:
            return
        self.child.update(targets)
        self.stage, self.message = self.child.stage, self.child.message
        self.side, self.tube = self.child.side, self.child.tube
        if self.phase == 'left_lift_bottle' and self.child.active:
            if self._bottle_parked():
                self._start_hold_mug()
            return
        if self.phase == 'hold_mug' and self.child.active:
            if self._mug_settled_at_hold_point():
                self._start_pour()
            return
        if self.phase == 'pour' and self.child.active and self.child.stage == 'await_cup':
            self._start_present_mug()
            return
        if self.phase == 'present_mug' and self.child.active:
            if self._mug_settled_at_hold_point():
                self._start_pour_hold()
            return
        if self.phase == 'pour' and self.child.active and self.child.stage == 'await_cup_clear':
            self._start_retract_mug()
            return
        if self.phase == 'retract_mug' and self.child.active:
            if self._mug_settled_at_hold_point():
                self._start_pour_return()
            return
        if self.child.active:
            return
        if self.child.status != 'succeeded':
            self._finish(self.child.status, self.child.message, True)
            return
        if self.phase == 'bottle_relay':
            self._start_left_lift_bottle()
        elif self.phase == 'pour':
            self.mug_child.cleared_to_place = True
            self.mug_child.stage_started = float(self.data.time)
            self.child = self.mug_child
            self.phase = 'place_mug'
        elif self.phase == 'place_mug':
            self._finish('succeeded', 'Poured water: right arm held the mug, left arm poured from the bottle, both objects returned.')

    def apply_gripper_limit(self, targets):
        ctrl = self.child.apply_gripper_limit(targets) if self.child else targets.copy()
        if self.phase in ('hold_mug', 'present_mug', 'retract_mug') and getattr(self, 'bottle_child', None):
            ctrl = self.bottle_child.apply_gripper_limit(ctrl)
        if self.phase == 'pour' and getattr(self, 'mug_child', None):
            ctrl = self.mug_child.apply_gripper_limit(ctrl)
        return ctrl

    def cancel(self, targets):
        if self.active:
            self.child.cancel(targets)
            if self.phase == 'hold_mug' and getattr(self, 'bottle_child', None):
                self.bottle_child.cancel(targets)
            if self.phase == 'pour' and getattr(self, 'mug_child', None):
                self.mug_child.cancel(targets)
            self.status, self.message = self.child.status, self.child.message
            self.request_pause = True
            self.elapsed = float(self.data.time)-self.started

    def snapshot(self):
        result = self.child.snapshot() if self.child else super().snapshot()
        result.update(status=self.status, active=self.active, kind=self.kind, message=self.message,
                      elapsed_s=round(float(self.data.time)-self.started, 2) if self.active else getattr(self, 'elapsed', 0.),
                      phase=getattr(self, 'phase', None),
                      cooperation='The bottle is relayed right-to-left through a table-supported point; '
                                  'the right cup arm remains fixed while the left bottle arm pours; both return to '
                                  'their table settings.')
        return result
