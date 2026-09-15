"""Contact-only dinner manipulation teacher using the shared IK and trajectory checks."""
import math
import mujoco
import numpy as np
from .autonomy import LiftReturn, ArmIK, PlanningError, OPEN
from .scene import HOME

CLOSED = -.170


DINNER_LABELS = {
    'idle': 'Ready for a dinner goal', 'planning': 'Check reach and clearance',
    'approach': 'Approach the object', 'descend': 'Align the fingers',
    'close': 'Grasp and check both fingers', 'retry': 'Reopen for one grasp retry',
    'lift': 'Lift the object', 'hold': 'Verify an unsupported hold',
    'clearance': 'Withdraw from the drawer', 'transit': 'Carry clear of the drawer',
    'rotate': 'Orient the held object for placement', 'align': 'Move over the placement guide',
    'lower': 'Lower to measured table support', 'release': 'Release the object',
    'retract': 'Withdraw the gripper', 'park': 'Park the working arm',
    'verify': 'Verify stable placement',
}
SKILLS = ('bottle', 'plate', 'mug', 'drawer', 'fork', 'spoon')


class DinnerTask(LiftReturn):
    def __init__(self, model, data, layout):
        super().__init__(model, data, layout)
        self.open_grip = OPEN
        self.message = 'Choose a dinner goal. Uses exact simulator state and physical contacts.'

    def snapshot(self):
        result = super().snapshot()
        result['stage_label'] = DINNER_LABELS.get(self.stage, self.stage)
        result['stages'] = [{'id': s, 'label': DINNER_LABELS.get(s,s)} for s in self.stages]
        result['object_id'] = result['tube_id']
        if self.kind == 'drawer_open':
            result['stage_label'] = 'Pull the passive drawer' if self.stage == 'align' else result['stage_label']
        return result

    def _stage(self, stage, message=None):
        super()._stage(stage, message or DINNER_LABELS.get(stage,stage))

    def start(self, side="auto", object_id="bottle", **kwargs):
        if self.active:
            raise ValueError("Cancel the current goal first.")
        if object_id not in ('bottle','plate','mug','fork','spoon'):
            raise ValueError("Unknown dinner object.")
        self.__init__(self.model, self.data, self.layout)
        self.requested_side, self.requested_object = side, object_id
        self.kind = "dinner_place"
        if object_id in ('fork','spoon'):
            self.stages[6:6] = ['clearance','transit'] + (['rotate'] if object_id == 'spoon' else [])
        self.status = "running"
        self._stage("planning")

    def _select_item(self, side, item):
        self.side, self.tube = side, item
        self.grip_torque = .5 if item["id"] == "plate" else .35 if item["id"] == "mug" else .3 if item["id"] in ("fork","spoon") else .15
        self.offset = 0 if side == "left" else 6
        self.ik = ArmIK(self.model, self.data, side)
        self.ik.axis_index = 1 if item['id'] == 'bottle' else 2
        if self.ik.axis_index == 2:
            self.ik.grasp_point = np.array([.003, 0., -.100])
            self.ik.x_target = np.array([0., -1., 0.])
            if item["id"] == "plate":
                self.ik.x_target = np.array([np.cos(-1.0), np.sin(-1.0), 0.])
        self.body = self.model.body(item['body']).id
        self.base_geom = self.model.geom('table').id
        moving = self.model.body(side + '_moving_jaw_so101_v1').id
        self.arm_geoms, self.fixed_geoms, self.moving_geoms = set(), set(), set()
        for i in range(self.model.ngeom):
            body = int(self.model.geom_bodyid[i])
            name = self.model.geom(i).name or ''
            if self.model.body(body).name.startswith(side+'_'):
                self.arm_geoms.add(i)
            if name.startswith(side+'_fixed_jaw') or (body == self.ik.body and self.model.geom_group[i] == 4):
                self.fixed_geoms.add(i)
            if body == moving:
                self.moving_geoms.add(i)
        self.jaw_geoms = self.fixed_geoms | self.moving_geoms
        self.origin = self.data.xpos[self.body].copy()
        self.sideways = item['id'] == 'bottle' and self.data.xmat[self.body,8] < .5
        if self.sideways:
            self.grip_torque = .65
            self.open_grip = .85
            self.ik.grasp_point = np.array([.016,0.,-.092])
            self.ik.axis_target = self.data.xmat[self.body].reshape(3,3)[:,2].copy()
            if 'rotate' not in self.stages:
                self.stages.insert(self.stages.index('align'), 'rotate')
        self.parked = self.data.qpos[6:12].copy() if side == 'left' else self.data.qpos[:6].copy()
        self.source_slot = {'id': item['id']+'_start'}
        target = next((t for t in self.layout['targets'] if t['object_id'] == item['id']), None)
        self.destination_position = np.array(target['position_m']) if target else np.array(item.get("initial_position_m",self.origin)) + [.070, -.040, 0]
        self.destination_position[2] = self.layout['table_z'] if item['id'] in ('fork','spoon') else self.origin[2]
        if self.sideways:
            self.destination_position[2] = self.layout['table_z']
        self.destination = {'id': target['id'] if target else 'bottle_place', 'position_m': self.destination_position.tolist()}
        self.others = {o['id']: self.data.body(o['body']).xpos.copy() for o in self.layout['objects'] if o['id'] != item['id']}
        self.metrics = {'gripper_torque_limit_nm': self.grip_torque, 'other_arm_max_motion_deg': 0., 'hold_verified_s': 0., 'unexpected_collisions': 0}

    def _plan(self, targets):
        if np.max(np.abs(self.data.qpos[:12]-np.array(HOME*2))) > .10 or np.max(np.abs(self.data.qvel[:12])) > .12:
            raise PlanningError('Start with both arms parked.')
        item = next(o for o in self.layout['objects'] if o['id'] == self.requested_object)
        failures = []
        for side in (['left', 'right'] if self.requested_side == 'auto' else [self.requested_side]):
            try:
                self._select_item(side, item)
                self.grasp = self.data.site(item['grasp_site']).xpos.copy()
                if self.sideways:
                    self.grasp = self.origin + self.ik.axis_target*.070
                if item['id'] in ('fork','spoon'):
                    if self.data.joint('drawer_slide').qpos[0] < .105:
                        raise PlanningError('Open the drawer before retrieving cutlery.')
                    self.ik.x_target = np.array([-1.,0.,0.])
                    self.ik.grasp_point = np.array([-.003,0.,-.100])
                if item['id'] == 'mug':
                    self.ik.x_target = np.array([.55,-np.sqrt(1-.55**2),0.])
                    self.grasp = self.origin + self.ik.x_target*.023 + [0,0,.041]
                    self.ik.grasp_point = np.array([-.006,0.,-.092])
                initial = np.array(HOME[:5])
                if item["id"] == "mug":
                    initial[4] = -2.4
                if item["id"] == "plate":
                    initial[4] = -2.4
                    self.grasp = self.origin + self.ik.x_target*.061 + [0,0,.013]
                    self.ik.grasp_point = np.array([-.003,0.,-.100])
                q = self.ik.solve(self.grasp, initial)
                self.hover = self.grasp + [0, 0, .025 if item['id'] in ('fork','spoon') else .055]
                try:
                    above = self.ik.solve(self.hover, q)
                except PlanningError:
                    if item['id'] != 'mug':
                        raise
                    self.hover = self.grasp + [0,0,.040]
                    above = self.ik.solve(self.hover,q)
                    self.metrics['approach_alternative'] = '40 mm hover'
                if item['id'] == 'spoon':
                    front = self.hover + [0,-.04,0]
                    qfront = self.ik.solve(front, above)
                    approach = np.vstack((self._joint_path(self.data.qpos[self.offset:self.offset+5], qfront, self.open_grip), self._cartesian(front,self.hover,qfront,self.open_grip)))
                else:
                    approach = self._joint_path(self.data.qpos[self.offset:self.offset+5], above, self.open_grip)
                self._cartesian(self.hover, self.grasp, above, self.open_grip)
                self.attempts = 1
                targets[:] = HOME*2
                self._move('approach', approach, self.open_grip, 3.)
                return
            except PlanningError as exc:
                failures.append(side+': '+str(exc))
        raise PlanningError('; '.join(failures))

    def _observe(self):
        force = {'fixed': 0., 'moving': 0., 'external': 0., 'base': 0.}
        wrench = np.zeros(6)
        for i, c in enumerate(self.data.contact):
            a, b = int(c.geom1), int(c.geom2)
            if self.model.geom_bodyid[a] != self.body and self.model.geom_bodyid[b] != self.body:
                continue
            other = b if self.model.geom_bodyid[a] == self.body else a
            mujoco.mj_contactForce(self.model, self.data, i, wrench)
            f = max(0., float(wrench[0]))
            key = 'fixed' if other in self.fixed_geoms else 'moving' if other in self.moving_geoms else 'external'
            force[key] += f
            if other == self.base_geom:
                force['base'] += f
        lift = float(self.data.xpos[self.body, 2]-self.origin[2])
        up = float(self.data.xmat[self.body, 8])
        both = min(force['fixed'], force['moving']) > .08
        self.max_lift = max(lift, self.max_lift)
        self.metrics.update(lift_cm=lift*100, max_lift_cm=self.max_lift*100, both_fingers=both,
                            finger_forces_n=[force['fixed'], force['moving']], external_contact_n=force['external'],
                            tilt_deg=float(np.rad2deg(np.arccos(np.clip(up, -1, 1)))))
        parked = self.data.qpos[6:12] if self.side == 'left' else self.data.qpos[:6]
        self.metrics['other_arm_max_motion_deg'] = max(self.metrics['other_arm_max_motion_deg'], float(np.rad2deg(np.max(np.abs(parked-self.parked)))))
        self.metrics['peak_tilt_deg'] = max(self.metrics.get('peak_tilt_deg',0.), self.metrics['tilt_deg'])
        return force, lift, up, both

    def _move_point(self, stage, end, grip, duration):
        points = self._cartesian(self.ik.point(self.data), np.asarray(end), self.data.qpos[self.offset:self.offset+5], grip, check=False)
        carry = self._carry_reference() if self.kind != 'drawer_open' and stage in ('lift','clearance','transit','align','lower') else None
        actual_grip = float(self.data.qpos[self.offset+5]) if carry is not None else grip
        if stage == 'lift' and self.sideways:
            # Permit initial table support only at departure; the remaining
            # hypothetical carried path must clear the table normally.
            self._check_path(points[:2], actual_grip, True, carry=carry, support=self.base_geom)
            self._check_path(points[2:], actual_grip, True, carry=carry)
        else:
            self._check_path(points, actual_grip, allow_tube=True, carry=carry, support=self.base_geom if stage == 'lower' else None)
        self._move(stage, points, grip, duration)

    def update(self, targets):
        if not self.active:
            return
        try:
            if self.stage == 'planning':
                self._plan(targets)
                return
            now = float(self.data.time)
            if now-self.started > 90 or now-self.stage_started > self.motion.duration+5:
                raise PlanningError('Motion timed out: '+self.stage)
            targets[self.offset:self.offset+6] = self.motion.sample(now)
            forces, lift, up, both = self._observe()
            collision = self._collision(self.data, True, .0008)
            if collision:
                self.metrics['unexpected_collisions'] += 1
                raise PlanningError('Unexpected contact: '+collision)
            if self.metrics['other_arm_max_motion_deg'] > 1.:
                raise PlanningError('The parked arm moved unexpectedly.')
            if self.stage in ('lift', 'hold', 'clearance', 'transit', 'rotate', 'align', 'lower') and lift > .012:
                if not both:
                    self.bad_grip_since = self.bad_grip_since or now
                    if now-self.bad_grip_since > .18:
                        raise PlanningError('Lost verified finger contact.')
                else:
                    self.bad_grip_since = None
                if not self.sideways and up < math.cos(math.radians(30 if self.tube["id"] in ("fork","spoon") else 15)):
                    raise PlanningError('Object exceeded its allowed carrying tilt.')
            if self.kind != 'drawer_open' and self.stage in ('clearance','transit','rotate','align') and forces['external'] > .10:
                raise PlanningError('The carried object contacted external support.')
            done = self._done_motion()
            if self.stage == 'approach' and done:
                self._move_point('descend', self.grasp, self.open_grip, 3.)
            elif self.stage == 'descend' and done:
                q = self.data.qpos[self.offset:self.offset+5].copy()
                self._move('close', np.array([q,q]), CLOSED, 5.)
            elif self.stage == 'close' and done:
                if not both or (not self.sideways and up < math.cos(math.radians(10))):
                    if self.attempts < 2 and np.linalg.norm(self.data.xpos[self.body]-self.origin) < .006:
                        self.attempts += 1
                        q = self.data.qpos[self.offset:self.offset+5].copy()
                        self._move('retry',np.array([q,q]),self.open_grip,4.)
                        return
                    raise PlanningError('Both fingers did not establish a grasp after the allowed attempts.')
                endpoint = self.ik.point(self.data)+[0,-.020 if self.tube['id']=='spoon' else 0,.07 if self.tube['id']=='bottle' else .055 if self.tube['id']=='plate' else .035]
                if self.sideways:
                    # First withdraw vertically; reorientation also translates
                    # toward the shared workspace to avoid a wrist singularity.
                    endpoint[2] = self.ik.point(self.data)[2] + .055
                try:
                    self._move_point('lift',endpoint,CLOSED,3.5)
                except PlanningError as exc:
                    if self.tube['id'] != 'fork' or 'useful reach' not in str(exc):
                        raise
                    start = self.ik.point(self.data).copy()
                    middle = start+[0,0,.020]
                    first = self._cartesian(start,middle,self.data.qpos[self.offset:self.offset+5],CLOSED,check=False)
                    second = self._cartesian(middle,endpoint+[0,-.020,0],first[-1],CLOSED,check=False)
                    points = np.vstack((first,second))
                    self._check_path(points,float(self.data.qpos[self.offset+5]),True,carry=self._carry_reference())
                    self._move('lift',points,CLOSED,4.)
                    self.metrics['lift_alternative'] = '20 mm vertical, then forward and upward'
            elif self.stage == 'retry' and done:
                self._move_point('descend',self.grasp,self.open_grip,2.)
            elif self.stage == 'lift' and done:
                if not (both and lift > (.05 if self.tube['id']=='bottle' else .02) and forces['external'] < .02):
                    raise PlanningError('Lift lacks independent finger support.')
                q = self.data.qpos[self.offset:self.offset+5].copy()
                self._move('hold', np.array([q,q]), CLOSED, 1.5)
            elif self.stage == 'hold':
                if not both or lift < (.05 if self.tube['id']=='bottle' else .02) or forces['external'] >= .02:
                    raise PlanningError('Hold slipped or regained external support.')
                self.metrics['hold_verified_s'] = now-self.stage_started
                if done:
                    if self.sideways:
                        q = self.data.qpos[self.offset:self.offset+5].copy()
                        reference = self._carry_reference()
                        # Express the physically held bottle axis in the wrist frame.
                        # This changes an IK constraint, never the object's pose.
                        self.ik.axis_local = reference[1][:,2].copy()
                        self.ik.axis_target = np.array([0.,0.,1.])
                        end = self.ik.solve(np.array([.025,-.075,self.layout['table_z']+.190]),np.array(HOME[:5]))
                        points = np.linspace(q,end,100)
                        self._check_path(points,float(self.data.qpos[self.offset+5]),True,carry=reference)
                        self._move('rotate',np.array(points),CLOSED,5.)
                        return
                    if self.tube['id'] in ('fork','spoon'):
                        self._move_point('clearance', self.ik.point(self.data)+[0,-.050-self.ik.point(self.data)[1],0], CLOSED, 4.)
                        return
                    center = self.destination_position.copy()
                    center[2] = self.destination_position[2]+.03 if self.tube["id"] == "plate" else self.data.xpos[self.body,2]
                    point, _ = self._point_for_center(center, self._carry_reference(), self.data.qpos[self.offset:self.offset+5])
                    self._move_point('align', point, CLOSED, 4.)
            elif self.stage == 'clearance' and done:
                center = self.data.xpos[self.body].copy()
                center[0] = -.10 if self.tube["id"] == "spoon" else self.destination_position[0]
                point, _ = self._point_for_center(center, self._carry_reference(), self.data.qpos[self.offset:self.offset+5])
                self._move_point('transit', point, CLOSED, 4.)
            elif self.stage == 'transit' and done and self.tube['id'] == 'spoon':
                point = self.ik.point(self.data).copy()
                q = self.data.qpos[self.offset:self.offset+5].copy()
                reference = self._carry_reference()
                points = []
                for angle in np.linspace(math.pi, math.pi/2, 40):
                    self.ik.x_target = np.array([np.cos(angle),np.sin(angle),0.])
                    q = self.ik.solve(point,q)
                    points.append(q.copy())
                self._check_path(points,float(self.data.qpos[self.offset+5]),True,carry=reference)
                self._move('rotate',np.array(points),CLOSED,4.)
            elif self.stage in ('transit','rotate') and done:
                if self.sideways:
                    if up < .98:
                        raise PlanningError('Bottle failed to become upright after reorientation.')
                    self.sideways = False
                center = self.destination_position.copy()
                center[2] = self.destination_position[2]+.03 if self.tube["id"] == "plate" else self.data.xpos[self.body,2]
                point, _ = self._point_for_center(center, self._carry_reference(), self.data.qpos[self.offset:self.offset+5])
                self._move_point('align', point, CLOSED, 4.)
            elif self.stage == 'align' and done:
                p = self.ik.point(self.data).copy()
                p[2] += self.destination_position[2]-.0005-self.data.xpos[self.body,2]
                self._move_point('lower', p, CLOSED, 4.)
            elif self.stage == 'lower':
                if forces['base'] > .06 and abs(self.data.xpos[self.body,2]-self.destination_position[2]) < .003:
                    p = self.ik.point(self.data)-self.data.xmat[self.ik.body].reshape(3,3)[:,0]*.010
                    self._move_point('release', p, self.open_grip, 4.)
                elif done:
                    raise PlanningError('No measured table support; refusing release.')
            elif self.stage == 'release' and done:
                endpoint = self.ik.point(self.data)+[0,0,.065 if self.tube['id'] in ('bottle','fork','spoon') else .035]
                try:
                    self._move_point('retract',endpoint,self.open_grip,3.)
                except PlanningError as exc:
                    if self.tube['id'] != 'spoon' or 'useful reach' not in str(exc):
                        raise
                    self._move_point('retract',endpoint-[0,0,.010],self.open_grip,3.)
                    self.metrics['retract_alternative'] = '55 mm retreat'
            elif self.stage == 'retract' and done:
                path = self._joint_path(self.data.qpos[self.offset:self.offset+5], np.array(HOME[:5]), self.open_grip)
                self._move('park', path, HOME[5], 3.)
            elif self.stage == 'park' and done:
                q = self.data.qpos[self.offset:self.offset+5].copy()
                self._move('verify', np.array([q,q]), HOME[5], 1.)
            elif self.stage == 'verify':
                joint = self.model.joint(self.tube['body']+'_free').dofadr[0]
                error = float(np.linalg.norm(self.data.xpos[self.body,:2]-self.destination_position[:2]))
                disturbed = max((float(np.linalg.norm(self.data.body(name).xpos-pos)) for name,pos in self.others.items()),default=0.)
                z_error = abs(float(self.data.xpos[self.body,2]-self.destination_position[2]))
                speed = float(np.linalg.norm(self.data.qvel[joint:joint+3]))
                angular = float(np.linalg.norm(self.data.qvel[joint+3:joint+6]))
                yaw_error = 0.
                if self.tube['id'] in ('fork','spoon'):
                    rot = self.data.xmat[self.body].reshape(3,3)
                    yaw = math.atan2(rot[1,0],rot[0,0])
                    desired = -math.pi/2 if self.tube['id']=='spoon' else 0.
                    yaw_error = abs(math.atan2(math.sin(yaw-desired),math.cos(yaw-desired)))
                valid = error < .006 and z_error < .003 and up > .98 and forces['base'] > .06 and forces['fixed']+forces['moving'] < .01 and speed < .003 and angular < .08 and disturbed < .004 and yaw_error < .175
                self.metrics.update(placement_z_error_mm=z_error*1000, other_object_max_displacement_mm=disturbed*1000, placement_yaw_error_deg=math.degrees(yaw_error), placement_speed_mm_s=speed*1000)
                self.metrics['placement_xy_error_mm'] = error*1000
                if valid:
                    self.stable_since = self.stable_since or now
                    if now-self.stable_since > 1.:
                        self._finish('succeeded', self.tube['id']+' physically lifted, held, placed and released.')
                else:
                    self.stable_since = None
        except (PlanningError, np.linalg.LinAlgError) as exc:
            targets[:] = self.data.qpos[:12]
            self._finish('failed', str(exc).replace('carried tube','carried object')+' Physics paused for inspection.', True)





class DrawerTask(DinnerTask):
    def start(self, side='left', **kwargs):
        if self.active:
            raise ValueError('Cancel the current goal first.')
        self.__init__(self.model, self.data, self.layout)
        self.kind = 'drawer_open'
        self.status = 'running'
        self._stage('planning')

    def _plan(self, targets):
        if np.max(np.abs(self.data.qpos[:12]-np.array(HOME*2))) > .10 or np.max(np.abs(self.data.qvel[:12])) > .12:
            raise PlanningError('Start with both arms parked.')
        if self.data.joint('drawer_slide').qpos[0] > .008:
            raise PlanningError('The drawer must start closed.')
        self._select_item('left', {'id':'drawer', 'body':'cutlery_drawer'})
        self.ik.x_target = np.array([0.,1.,0.])
        self.ik.grasp_point = np.array([-.001,0.,-.096])
        self.grasp = self.data.site('drawer_handle_grasp').xpos.copy()
        self.hover = self.grasp + [0,0,.04]
        q = self.ik.solve(self.grasp, np.array(HOME[:5]))
        above = self.ik.solve(self.hover,q)
        front = self.hover + [0,-.05,0]
        qfront = self.ik.solve(front, above)
        approach = np.vstack((self._joint_path(self.data.qpos[:5], qfront, OPEN), self._cartesian(front,self.hover,qfront,OPEN)))
        self._cartesian(self.hover,self.grasp,above,OPEN)
        self.attempts = 1
        targets[:] = HOME*2
        self._move('approach',approach,OPEN,3.)

    def update(self, targets):
        if not self.active:
            return
        try:
            if self.stage == 'planning':
                self._plan(targets)
                return
            now = float(self.data.time)
            if now-self.started > 75 or now-self.stage_started > self.motion.duration+5:
                raise PlanningError('Drawer motion timed out: '+self.stage)
            targets[self.offset:self.offset+6] = self.motion.sample(now)
            forces, _, _, both = self._observe()
            opening = float(self.data.joint('drawer_slide').qpos[0])
            self.metrics['drawer_open_m'] = opening
            collision = self._collision(self.data,True,.0008)
            if collision:
                self.metrics['unexpected_collisions'] += 1
                raise PlanningError('Unexpected contact: '+collision)
            if self.metrics['other_arm_max_motion_deg'] > 1.:
                raise PlanningError('The parked arm moved unexpectedly.')
            if self.kind != 'drawer_open' and self.stage in ('clearance','transit','rotate','align') and forces['external'] > .10:
                raise PlanningError('The carried object contacted external support.')
            done = self._done_motion()
            if self.stage == 'approach' and done:
                self._move_point('descend',self.grasp,OPEN,3.)
            elif self.stage == 'descend' and done:
                q = self.data.qpos[:5].copy()
                self._move('close',np.array([q,q]),CLOSED,4.)
            elif self.stage == 'close' and done:
                if not both:
                    raise PlanningError('Drawer handle grasp was not verified at both fingers.')
                self._move_point('align',self.ik.point(self.data)+[0,-.116,0],CLOSED,6.)
                self.message = 'Pull the passive drawer using the gripper.'
            elif self.stage == 'align' and done:
                if opening < .105 or not both:
                    raise PlanningError('The drawer did not follow the verified handle grasp.')
                q = self.data.qpos[:5].copy()
                self._move('release',np.array([q,q]),OPEN,3.)
            elif self.stage == 'release' and done:
                self._move_point('retract',self.ik.point(self.data)+[0,0,.04],OPEN,3.)
            elif self.stage == 'retract' and done:
                self._move('park',self._joint_path(self.data.qpos[:5],np.array(HOME[:5]),OPEN),HOME[5],3.)
            elif self.stage == 'park' and done:
                if opening < .105:
                    raise PlanningError('Drawer did not remain open after release.')
                self._finish('succeeded','Drawer physically opened and released; arm parked.')
        except (PlanningError,np.linalg.LinAlgError) as exc:
            targets[:] = self.data.qpos[:12]
            self._finish('failed',str(exc).replace('carried tube','carried object')+' Physics paused for inspection.',True)








class DinnerSequence(LiftReturn):
    """Orchestrate independently verified skills without changing physical state."""
    def __init__(self, model, data, layout):
        super().__init__(model,data,layout)
        self.child = None
        self.steps, self.results = [], []
        self.kind = 'set_table'
        self.message = 'Ready: run the dinner sequence or choose one physical skill.'
        self.final_positions = {}

    def start(self, side='auto', object_id=None, kind='set_table', **kwargs):
        if self.active:
            raise ValueError('A goal is already running. Cancel it first.')
        if self.layout['dinner_preset'] != 'task':
            raise ValueError('Load Task start before running a dinner goal.')
        if kind not in ('set_table','dinner_place','drawer_open'):
            raise ValueError('Choose a dinner goal for this scene.')
        if side not in ('auto','left','right'):
            raise ValueError('Unknown arm.')
        if kind == 'dinner_place' and object_id not in ('bottle','plate','mug','fork','spoon'):
            raise ValueError('Choose bottle, plate, mug, fork or spoon.')
        if kind == 'set_table' and side != 'auto':
            raise ValueError('The full sequence assigns both arms automatically.')
        if kind == 'drawer_open' and side == 'right':
            raise ValueError('The drawer skill currently uses the left arm.')
        if kind == 'set_table' and self.data.joint('drawer_slide').qpos[0] > .008:
            raise ValueError('Run all six skills requires the closed Task start setup.')
        self.__init__(self.model,self.data,self.layout)
        self.kind, self.requested_side = kind,side
        self.steps = list(SKILLS) if kind == 'set_table' else ['drawer' if kind == 'drawer_open' else object_id]
        self.status = 'running'
        self._next()

    def _next(self):
        name = self.steps[len(self.results)]
        self.child = DrawerTask(self.model,self.data,self.layout) if name == 'drawer' else DinnerTask(self.model,self.data,self.layout)
        self.child.start(side=self.requested_side,object_id=name)
        self.stage = self.child.stage

    def apply_gripper_limit(self, targets):
        return self.child.apply_gripper_limit(targets) if self.child else targets.copy()

    def cancel(self, targets):
        if self.active:
            self.child.cancel(targets)
            self.status, self.message = self.child.status,self.child.message
            self.request_pause = True
            self.elapsed = float(self.data.time)-self.started

    def update(self, targets):
        if not self.active:
            return
        self.child.update(targets)
        self.stage, self.message = self.child.stage,self.child.message
        self.side, self.tube = self.child.side,self.child.tube
        if self.child.active:
            return
        if self.child.status != 'succeeded':
            self._finish(self.child.status,self.child.message,True)
            return
        name = self.steps[len(self.results)]
        self.results.append({'skill':name, **self.child.snapshot()})
        if name != 'drawer':
            self.final_positions[name] = self.data.body(name).xpos.copy()
        if len(self.results) < len(self.steps):
            self._next()
            return
        disturbed = max((float(np.linalg.norm(self.data.body(name).xpos-pos)) for name,pos in self.final_positions.items()),default=0.)
        if disturbed > .004:
            self._finish('failed','A later skill disturbed an earlier placement. Physics paused for inspection.',True)
        elif 'drawer' in self.steps and self.data.joint('drawer_slide').qpos[0] < .105:
            self._finish('failed','The drawer did not remain open. Physics paused for inspection.',True)
        else:
            self._finish('succeeded',f'Completed {len(self.steps)} physical skill(s). Objects released and both arms parked.')

    def snapshot(self):
        result = self.child.snapshot() if self.child else super().snapshot()
        result.update(status=self.status,active=self.active,kind=self.kind,message=self.message,
                      elapsed_s=round(float(self.data.time)-self.started,2) if self.active else getattr(self,'elapsed',0.),
                      recording_id=self.recording_id,steps=self.steps,completed_steps=[r['skill'] for r in self.results],
                      results=list(self.results),observation='exact_simulator_state',attachment='physical_contacts_only')
        if self.steps:
            result['progress'] = 1. if self.status=='succeeded' else min(.999,(len(self.results)+(result['progress'] if self.child and self.child.active else 0))/len(self.steps))
            result['stage_label'] = f"{self.steps[min(len(self.results),len(self.steps)-1)].title()} · {result['stage_label']}"
        return result
