"""Privileged physical scoring for learned dinner actions and saved-action replay.

This module never generates motor targets. It may stop an unsafe trial and score
its outcome; no object positions or teacher stages are returned to a policy.
"""
import math
import mujoco
import numpy as np
from .dinner_autonomy import DinnerTask
from .scene import HOME


class DinnerPhysicalMonitor:
    def __init__(self, model, data, layout, skill, side):
        self.model, self.data, self.skill, self.side = model, data, skill, side
        self.observer = DinnerTask(model, data, layout)
        item = ({'id': 'drawer', 'body': 'cutlery_drawer'} if skill == 'drawer'
                else next(o for o in layout['objects'] if o['id'] == skill))
        self.observer._select_item(side, item)
        self.destination = self.observer.destination_position.copy()
        self.previous_time = float(data.time)
        self.held = self.best_hold = self.gap = self.longest_gap = self.stable = 0.
        self.max_other = self.max_parked = self.handle_hold = self.max_opening = 0.
        self.collisions = 0
        self.failure = None
        self.metrics = {}

    def update(self):
        d, o = self.data, self.observer
        dt = max(0., float(d.time) - self.previous_time)
        self.previous_time = float(d.time)
        forces, lift, up, both = o._observe()
        collision = o._collision(d, allow_tube=True, penetration=.0008)
        self.collisions += int(bool(collision))
        displacements={n:float(np.linalg.norm(d.body(n).xpos-p)) for n,p in o.others.items()}
        other = max(displacements.values(), default=0.)
        # Cutlery moves normally with the unactuated drawer. Other objects do not.
        if self.skill == 'drawer':
            other = max((float(np.linalg.norm(d.body(n).xpos-p)) for n, p in o.others.items()
                         if n not in ('fork', 'spoon')), default=0.)
        self.max_other = max(self.max_other, other)
        self.max_parked = max(self.max_parked, o.metrics['other_arm_max_motion_deg'])
        parked = (np.max(np.abs(d.qpos[:12]-np.array(HOME*2))) < .035
                  and np.max(np.abs(d.qvel[:12])) < .12)
        if self.skill == 'drawer':
            opening = float(d.joint('drawer_slide').qpos[0])
            self.max_opening = max(self.max_opening, opening)
            if both: self.handle_hold += dt
            valid = (opening >= .105 and self.handle_hold >= .1
                     and forces['fixed']+forces['moving'] < .02 and parked)
            self.metrics.update(drawer_open_m=opening, max_opening_m=self.max_opening,
                                verified_handle_contact_s=self.handle_hold)
        else:
            threshold = .05 if self.skill == 'bottle' else .02
            self.held = self.held+dt if both and lift > threshold and forces['external'] < .02 else 0.
            self.best_hold = max(self.best_hold, self.held)
            self.gap = self.gap+dt if lift > .012 and not both and forces['base'] < .02 else 0.
            self.longest_gap = max(self.longest_gap, self.gap)
            body = d.body(self.skill)
            error = float(np.linalg.norm(body.xpos[:2]-self.destination[:2]))
            z_error = abs(float(body.xpos[2]-self.destination[2]))
            dof = self.model.joint(self.skill+'_free').dofadr[0]
            speed = float(np.linalg.norm(d.qvel[dof:dof+3]))
            angular = float(np.linalg.norm(d.qvel[dof+3:dof+6]))
            yaw_error = 0.
            if self.skill in ('fork', 'spoon'):
                rotation = body.xmat.reshape(3, 3)
                yaw = math.atan2(rotation[1, 0], rotation[0, 0])
                desired = -math.pi/2 if self.skill == 'spoon' else 0.
                yaw_error = abs(math.atan2(math.sin(yaw-desired), math.cos(yaw-desired)))
            valid = (self.best_hold >= 1.49 and error < .008 and z_error < .004 and up > .98
                     and speed < .003 and angular < .08 and yaw_error < .175
                     and forces['base'] > .02 and forces['fixed']+forces['moving'] < .02 and parked)
            self.metrics.update(placement_error_mm=error*1000, placement_z_error_mm=z_error*1000,
                                placement_yaw_error_deg=math.degrees(yaw_error), speed_mm_s=speed*1000,
                                max_lift_cm=o.max_lift*100, hold_verified_s=self.best_hold,
                                longest_unsupported_gap_s=self.longest_gap)
        self.stable = self.stable+dt if valid else 0.
        self.metrics.update(stable_release_and_park_s=self.stable, both_arms_parked=bool(parked),
                            other_object_displacements_m=displacements,
                            other_object_max_displacement_m=self.max_other,
                            parked_arm_max_motion_deg=self.max_parked, unexpected_collisions=self.collisions)
        if collision: self.failure = 'Unexpected arm contact: '+collision
        elif self.max_other > .004: self.failure = 'Another object was disturbed.'
        elif self.max_parked > 1.: self.failure = 'The parked arm moved unexpectedly.'
        elif self.longest_gap > .18: self.failure = 'The carried object lost finger support.'
        return self.failure

    @property
    def succeeded(self):
        return self.failure is None and self.stable >= .5

    def report(self):
        return {'passed': bool(self.succeeded), 'failure': self.failure, 'metrics': dict(self.metrics),
                'teacher_updates': 0, 'scoring': 'Privileged simulator state; no motor targets generated'}
