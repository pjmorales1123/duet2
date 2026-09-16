"""Experimental image-feedback bottle controller with explicit task phases.

RGB estimates locate the grasp and update its approach/descent. After closure,
robot proprioception and the observed grasp offset define a Cartesian carry.
This first candidate does not visually replan carried-object paths. A caller's
separate physical monitor must stop/score every trial. No simulator data object,
object pose, teacher, IK solver or saved trajectory is accepted by this class.
"""
import numpy as np
from .scene import HOME


def smooth(t):
    t = np.clip(t, 0., 1.)
    return t**3*(10.+t*(-15.+6.*t))


class RgbServoBottle:
    durations = {'approach': 3., 'descend': 3., 'close': 5., 'lift': 3.5, 'hold': 1.8,
                 'align': 4., 'lower': 4., 'release': 4., 'retract': 3., 'park': 3., 'verify': 2.}
    fixed_jaw_clearance_m = .004

    def __init__(self, motor, destination, table_z=.76, image_mode='live'):
        if image_mode not in ('live', 'frozen'):
            raise ValueError('Choose live or frozen initial images.')
        self.motor = motor; self.destination = np.asarray(destination, dtype=float)
        if self.destination.shape != (2,) or not np.isfinite(self.destination).all():
            raise ValueError('A finite planar destination is required.')
        self.table_z = float(table_z); self.image_mode = image_mode
        self.stage = 'observe'; self.status = 'running'; self.side = None
        self.observation = None; self.last_observation_time = None; self.history = []
        self.initial_observation = None
        self.current_rgb_estimate = None
        self.last_target = np.asarray(HOME*2, dtype=float)
        self.refusals = 0; self.observations = 0; self.message = ''
        self.empty_gripper_since = None

    def accept_observation(self, observation, now):
        if observation['status'] != 'observed':
            if self.image_mode == 'live' or self.observation is None:
                self.refusals += 1
            return
        point = np.asarray(observation['grasp_point_m'], dtype=float)
        if point.shape != (3,) or not np.isfinite(point).all():
            raise ValueError('Invalid RGB grasp point.')
        self.current_rgb_estimate = (point.copy() if self.current_rgb_estimate is None else
                                     self.current_rgb_estimate+.35*(point-self.current_rgb_estimate))
        if self.image_mode == 'frozen' and self.observation is not None:
            return
        self.observation = self.current_rgb_estimate.copy(); self.last_observation_time = float(now); self.observations += 1
        if self.initial_observation is None:
            self.initial_observation = point.copy()

    def preview_image_action(self, point, now, positions):
        """Same-proprioception counterfactual; never advances the controller."""
        if self.stage not in ('approach', 'descend'):
            return None
        fraction = smooth((now-self.stage_started)/self.durations[self.stage])
        point = self.grasp_goal(self.side, point)
        if self.stage == 'approach':
            end = self.motor.predict(self.side, np.asarray(point)+[0., 0., .055])
            desired = self.qstart+(end-self.qstart)*fraction
        else:
            desired = self.motor.predict(self.side, self.pstart+(np.asarray(point)-self.pstart)*fraction)
        measured = np.asarray(positions)[self.offset:self.offset+5]
        return measured+np.clip(desired-measured, -.08, .08)

    def grasp_goal(self, side, point):
        point = np.asarray(point, dtype=float)
        neutral = self.motor.predict(side, point)
        _, rotation = self.motor.geometry.pose(side, neutral)
        return point-rotation[:, 0]*self.fixed_jaw_clearance_m

    def enter(self, stage, now, q):
        self.stage = stage; self.stage_started = float(now)
        self.qstart = q[self.offset:self.offset+5].copy()
        self.pstart, rotation = self.motor.geometry.pose(self.side, self.qstart)
        self.gripstart = float(self.last_target[self.offset+5])
        self.history.append({'stage': stage, 'time_s': float(now)})
        if stage == 'lift':
            # An image-derived offset is measured once at closure. This is not
            # an attachment: subsequent object motion is purely physical.
            self.grasp_offset = self.observation-self.pstart
            self.fixed_end = self.pstart+[0., 0., .070]
        elif stage == 'hold':
            self.fixed_end = self.pstart.copy()
        elif stage == 'align':
            self.fixed_end = np.r_[self.destination, self.table_z+.198]-self.grasp_offset
        elif stage == 'lower':
            self.fixed_end = np.r_[self.destination, self.table_z+.1275]-self.grasp_offset
        elif stage == 'release':
            self.fixed_end = self.pstart-rotation[:, 0]*.010
        elif stage == 'retract':
            self.fixed_end = self.pstart+[0., 0., .065]

    def choose_side(self, q):
        errors = []
        for side in ('left', 'right'):
            try:
                grasp_goal = self.grasp_goal(side, self.observation)
                # All endpoint checks use the learned map plus robot-only FK.
                for point in (grasp_goal, grasp_goal+[0, 0, .055],
                              grasp_goal+[0, 0, .070], np.r_[self.destination, self.table_z+.198],
                              np.r_[self.destination, self.table_z+.1275]):
                    self.motor.predict(side, point)
                self.side = side; self.offset = 0 if side == 'left' else 6
                return
            except ValueError as exc:
                errors.append(side+': '+str(exc))
        raise ValueError('; '.join(errors))

    def update(self, now, positions, velocities):
        q = np.asarray(positions, dtype=float); v = np.asarray(velocities, dtype=float)
        if q.shape != (12,) or v.shape != (12,) or not np.isfinite(np.r_[q, v]).all():
            raise ValueError('Twelve finite measured robot positions and velocities are required.')
        if self.status != 'running':
            return self.last_target.copy()
        try:
            if self.stage == 'observe':
                if self.observation is None:
                    if now > 1.:
                        raise ValueError('No confident initial RGB estimate.')
                    return self.last_target.copy()
                self.choose_side(q); self.enter('approach', now, q)
            elapsed = now-self.stage_started
            if elapsed > self.durations[self.stage]+4.:
                raise ValueError('Motor progress timed out during '+self.stage+'.')
            if self.stage in ('approach', 'descend') and self.image_mode == 'live' and now-self.last_observation_time > .6:
                raise ValueError('Current RGB estimate was unavailable for more than 0.6 seconds.')
            if self.stage in ('lift', 'hold', 'align', 'lower'):
                if q[self.offset+5] < -.08:
                    self.empty_gripper_since = now if self.empty_gripper_since is None else self.empty_gripper_since
                    if now-self.empty_gripper_since > .2:
                        raise ValueError('Measured gripper closure indicates an empty grasp.')
                else:
                    self.empty_gripper_since = None
            fraction = smooth(elapsed/self.durations[self.stage])
            grip = -.17 if self.stage in ('close', 'lift', 'hold', 'align', 'lower') else .4
            if self.stage == 'verify':
                grip = HOME[5]
            if self.stage == 'approach':
                grasp_goal = self.grasp_goal(self.side, self.observation)
                final_q = self.motor.predict(self.side, grasp_goal+[0., 0., .055])
                desired_q = self.qstart+(final_q-self.qstart)*fraction
            elif self.stage == 'descend':
                grasp_goal = self.grasp_goal(self.side, self.observation)
                point = self.pstart+(grasp_goal-self.pstart)*fraction
                desired_q = self.motor.predict(self.side, point); final_q = self.motor.predict(self.side, grasp_goal)
            elif self.stage in ('close', 'hold', 'verify'):
                desired_q = final_q = self.qstart.copy()
            elif self.stage == 'park':
                final_q = np.asarray(HOME[:5]); desired_q = self.qstart+(final_q-self.qstart)*fraction; grip = HOME[5]
            else:
                point = self.pstart+(self.fixed_end-self.pstart)*fraction
                desired_q = self.motor.predict(self.side, point); final_q = self.motor.predict(self.side, self.fixed_end)
            # Measured motor progress bounds a stalled arm's commanded error.
            target = self.last_target.copy()
            target[self.offset:self.offset+5] = q[self.offset:self.offset+5]+np.clip(desired_q-q[self.offset:self.offset+5], -.08, .08)
            target[self.offset+5] = self.gripstart+(grip-self.gripstart)*fraction
            self.last_target = target
            settled = (np.max(np.abs(final_q-q[self.offset:self.offset+5])) < .02
                       and np.max(np.abs(v[self.offset:self.offset+5])) < .15)
            if elapsed >= self.durations[self.stage] and settled:
                stages = list(self.durations); index = stages.index(self.stage)
                if index == len(stages)-1:
                    self.status = 'completed'; self.message = 'Controller finished; physical monitor decides success.'
                else:
                    self.enter(stages[index+1], now, q)
            return target
        except ValueError as exc:
            self.status = 'refused'; self.message = str(exc); self.last_target = q.copy()
            return self.last_target.copy()
