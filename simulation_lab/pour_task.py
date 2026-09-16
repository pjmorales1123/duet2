"""Two-arm 'put water' skill with a table-supported cup relay.

The right arm sets the cup down at a shared relay point and parks; the left
arm regrips it and holds it for the right arm's bottle pour. Both objects then
return to their own table settings. Built from the existing physical
pick/place primitives: no airborne handoff or object-state writes.
"""
from copy import deepcopy
import math
import numpy as np
from .autonomy import PlanningError
from .carry_routing import plan_disc_detour, segment_clear_of_disc
from .dinner_autonomy import CLOSED, DinnerSequence, DinnerTask

CUP_RELAY_POINT = (.15, -.14)  # reachable by the donor and clear of the plate's rim
MUG_HOLD_XY = (.10, -.08)  # verified via full-physics sweep: right can carry-hold here without hitting a joint limit, and it's close enough to center that left's pour approach has ~0.58rad of joint-limit margin (vs. ~0 at the mug's natural far-side resting spot)
MUG_HOLD_HEIGHT = .05  # above the table - low enough to stay in reach, high enough to clear cabinet/plate fixtures while translating in
POUR_OFFSET = np.array([0., -.03, .12])  # beside/above the held mug - the bottle's grasp point sits near its neck, well above its base, so it needs real clearance above the mug's low hold height to keep the bottle's body off the table
POUR_TILT_RAD = .9  # ~50 degrees, tips the grasped neck toward the mug



def _route_around_discs(start, end, discs):
    """Route from start to end clearing every (center, radius) obstacle in
    discs. plan_disc_detour only routes around one obstacle at a time, so
    detour around each disc in turn, then re-validate every leg of the
    result against *all* discs - a detour built for the plate can still clip
    the camera box, and vice versa - inserting one more detour wherever it
    does, until a single pass leaves nothing to fix."""
    route = [start, end]
    for _ in range(len(discs) + 1):
        fixed = True
        next_route = [route[0]]
        for a, b in zip(route, route[1:]):
            leg = [a, b]
            for center, radius in discs:
                if not segment_clear_of_disc(leg[0], leg[-1], center, radius):
                    leg = plan_disc_detour(leg[0], leg[-1], center, radius)
                    fixed = False
                    break
            next_route.extend(leg[1:])
        route = next_route
        if fixed:
            return route
    raise ValueError('No planar route clears every obstacle.')


def _with_object_at_relay_point(layout, object_id, point):
    layout = deepcopy(layout)
    target = next(t for t in layout['targets'] if t['object_id'] == object_id)
    target['position_m'] = [point[0], point[1], layout['table_z']]
    return layout


class CupDonorTask(DinnerTask):
    """Right arm placing the cup at the verified shared relay point."""

    def _move_center(self, stage, center, grip, duration):
        if stage != 'align':
            super()._move_center(stage, center, grip, duration)
            return
        # The direct cup path crosses the two fixed cabinet ledges at x=.18.
        # Move rearward first, then across to the relay point, while retaining
        # the physically held cup's measured carry height.
        current_center = self.data.xpos[self.body].copy()
        relay_center = center.copy()
        relay_center[2] = self.destination_position[2] + .030
        waypoints = [np.array([current_center[0], -.120, current_center[2]]), relay_center]
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
        self._check_path(points, float(self.data.qpos[self.offset+5]), True, carry=reference)
        self._move('align', points, grip, duration)


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
        super()._move_center(stage, center, grip, duration)


class PourBottleTask(DinnerTask):
    """Picks the bottle from the relay point and lifts it, then PARKS nearby
    (small retreat, no ambitious reach) and waits - the mug hasn't moved into
    the shared pour station yet, and it's much too close to the relay point
    for both events to happen at once without the arms colliding. Once told
    (via cleared_to_pour) that the mug is in place, it routes laterally
    around the plate to it, tilts and pours, then returns to its own
    setting - same as before, just no longer racing the mug for the same
    patch of table."""
    cleared_to_pour = False

    def __init__(self, model, data, layout, mug_body_id=None):
        super().__init__(model, data, layout)
        self.mug_body_id = mug_body_id

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
        pour_point = self.data.xpos[self.mug_body_id].copy() + POUR_OFFSET
        start = self.ik.point(self.data).copy()
        # The bottle is grasped near its cap, so it hangs a long way below
        # the gripper - descending all the way to POUR_OFFSET's low height
        # (matched to the mug's low hold height) drags the bottle's base
        # into the table. Stay at the already-proven-safe transit height for
        # the whole lateral move; only x/y change here. The final approach
        # to the mug's actual rim height happens in the tilt sequence, which
        # already re-solves IK once it's hovering directly above the mug.
        pour_point = pour_point.copy()
        pour_point[2] = start[2]
        # A direct straight-line transit clips the plate's rim and the right
        # arm's wrist camera housing (now planted at the mug hold point).
        # Route laterally around both, at this same safe Z the whole way -
        # exactly the detour dinner_autonomy.py's _move_fork_around_plate
        # already uses for the fork, just chained over two obstacles.
        plate = next(item for item in self.layout['objects'] if item['id'] == 'plate')
        plate_center = self.data.xpos[self.model.body(plate['body']).id].copy()
        bottle_radius = self.tube['size_m'][0]/2
        plate_clearance = plate['size_m'][0]/2 + bottle_radius + .012
        # Read the camera box pose now, live off the right arm mid-hold - not
        # at reset/spawn, which is a different pose entirely.
        camera_gid = self.model.geom('right_camera_box2').id
        camera_center = self.data.geom_xpos[camera_gid].copy()
        camera_half_x, camera_half_y = self.model.geom_size[camera_gid][:2]
        camera_r = np.hypot(camera_half_x, camera_half_y)
        camera_clearance = camera_r + bottle_radius + .01
        route = _route_around_discs(
            start[:2], pour_point[:2],
            [(plate_center[:2], plate_clearance), (camera_center[:2], camera_clearance)],
        )
        reference = self._carry_reference()
        current_point = start
        q = self.data.qpos[self.offset:self.offset+5].copy()
        paths = []
        for xy in route[1:]:
            target = np.array([xy[0], xy[1], start[2]])
            path = self._cartesian(current_point, target, q, grip, check=False)
            paths.append(path)
            current_point, q = target, path[-1]
        points = np.vstack(paths)
        self._check_path(points, grip, True, carry=reference)
        self._move('pour_approach', points, grip, 4.)

    def update(self, targets):
        if self.stage in ('pour_approach', 'pour_tilt', 'pour_hold', 'pour_return') and self.active:
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
                self.ik.axis_target = np.array([math.sin(POUR_TILT_RAD), 0., math.cos(POUR_TILT_RAD)])
                end = self.ik.solve(point, q)
                path = np.linspace(q, end, 60)
                self._check_path(path, CLOSED, True, carry=self._carry_reference())
                self._move('pour_tilt', path, CLOSED, 2.5)
            elif self.stage == 'pour_tilt' and done:
                self._move('pour_hold', np.array([q, q]), CLOSED, 1.5)
            elif self.stage == 'pour_hold' and done:
                self.ik.axis_target = np.array([0., 0., 1.])
                end = self.ik.solve(point, q)
                path = np.linspace(q, end, 60)
                self._check_path(path, CLOSED, True, carry=self._carry_reference())
                self._move('pour_return', path, CLOSED, 2.5)
            elif self.stage == 'pour_return' and done:
                self.poured = True
                center, grip, duration = self._final_align
                self._move_center('align', center, grip, duration)
        except (PlanningError, np.linalg.LinAlgError) as exc:
            targets[:] = self.data.qpos[:12]
            self._finish('failed', str(exc)+' Physics paused for inspection.', True)


class PourWaterTask(DinnerSequence):
    """'Put water': relay the cup right -> left, pour right -> left, return."""

    HOLD_VERIFIED_S = 1.6

    def start(self, **kwargs):
        if self.active:
            raise ValueError('Cancel the current goal first.')
        if self.layout['dinner_preset'] != 'task':
            raise ValueError('Load Task start before running a dinner goal.')
        layout = _with_object_at_relay_point(self.layout, 'mug', CUP_RELAY_POINT)
        self.__init__(self.model, self.data, layout)
        self.kind = 'pour_water'
        self.status = 'running'
        self._start_cup_donor()

    def _start_cup_donor(self):
        self.phase = 'cup_relay'
        self.child = CupDonorTask(self.model, self.data, self.layout)
        self.child.start(side='right', object_id='mug')
        self.stage = self.child.stage

    def _start_left_hold_cup(self):
        self.phase = 'hold_mug'
        self.child = HoldTask(self.model, self.data, self.layout)
        self.child.hold_point = np.array([MUG_HOLD_XY[0], MUG_HOLD_XY[1], self.layout['table_z']+MUG_HOLD_HEIGHT])
        self.child.start(side='right', object_id='mug')
        self.stage = self.child.stage

    def _start_right_lift_bottle(self):
        self.phase = 'right_lift_bottle'
        self.mug_child = self.child
        self.child = PourBottleTask(self.model, self.data, self.layout)
        self.child.start(side='right', object_id='bottle')
        self.child.require_home_start = False
        self.child.other_arm_tolerance_deg = 180.
        self.stage = self.child.stage

    def _bottle_parked(self):
        return self.child.stage == 'align' and self.child.motion is not None and self.child._done_motion()

    def _start_pour(self):
        self.phase = 'pour'
        self.child.mug_body_id = self.mug_child.body
        self.child.cleared_to_pour = True
        self.child.other_arm_tolerance_deg = 180.  # the right arm is deliberately holding the mug, not parked
        try:
            self.child._begin_pour_approach(CLOSED)
        except (PlanningError, ValueError) as exc:
            self.child._finish('failed', str(exc)+' Physics paused for inspection.', True)
        self.stage = self.child.stage

    def _mug_settled_at_hold_point(self):
        return self.child.stage == 'align' and self.child.motion is not None and self.child._done_motion()

    def update(self, targets):
        if not self.active:
            return
        self.child.update(targets)
        self.stage, self.message = self.child.stage, self.child.message
        self.side, self.tube = self.child.side, self.child.tube
        if self.phase == 'hold_mug' and self.child.active:
            if self._mug_settled_at_hold_point():
                self._start_right_lift_bottle()
            return
        if self.phase == 'right_lift_bottle' and self.child.active:
            if self._bottle_parked():
                self._start_pour()
            return
        if self.child.active:
            return
        if self.child.status != 'succeeded':
            self._finish(self.child.status, self.child.message, True)
            return
        if self.phase == 'cup_relay':
            self._start_left_hold_cup()
        elif self.phase == 'pour':
            self.mug_child.cleared_to_place = True
            self.child = self.mug_child
            self.phase = 'place_mug'
        elif self.phase == 'place_mug':
            self._finish('succeeded', 'Poured water: right arm held the mug, left arm poured from the bottle, both objects returned.')

    def apply_gripper_limit(self, targets):
        ctrl = self.child.apply_gripper_limit(targets) if self.child else targets.copy()
        if self.phase == 'right_lift_bottle' and getattr(self, 'mug_child', None):
            ctrl = self.mug_child.apply_gripper_limit(ctrl)
        if self.phase == 'pour' and getattr(self, 'mug_child', None):
            ctrl = self.mug_child.apply_gripper_limit(ctrl)
        return ctrl

    def cancel(self, targets):
        if self.active:
            self.child.cancel(targets)
            if self.phase == 'right_lift_bottle' and getattr(self, 'mug_child', None):
                self.mug_child.cancel(targets)
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
                      cooperation='The cup is relayed right-to-left through a table-supported shared point; '
                                  'the left arm holds the cup while the right arm pours; both objects return to '
                                  'their table settings.')
        return result
