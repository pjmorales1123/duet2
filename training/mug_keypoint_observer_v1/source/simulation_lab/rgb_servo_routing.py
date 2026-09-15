"""Neural-map route selection; separate from the preserved V1 controller.

Only RGB-derived points, requested destinations, robot kinematics and measured
joints are inputs. A route check cannot establish collision-free physics.
"""
import numpy as np
from .rgb_servo_controller import RgbServoBottle


class RoutedRgbServoBottle(RgbServoBottle):
    def __init__(self, motor, destination, table_z, image_mode, routing, side=None):
        super().__init__(motor, destination, table_z, image_mode)
        self.routing = routing; self.forced_side = side
        self.route_details = None

    def check_line(self, side, first, last):
        for fraction in np.linspace(0., 1., self.routing['route_samples']):
            self.motor.predict(side, first+(last-first)*fraction)

    def transport(self, side, first, offset):
        lower = np.r_[self.destination, self.table_z+.1275]-offset
        errors = []
        for height in self.routing['transport_height_above_table_m']:
            end = np.r_[self.destination, self.table_z+height]-offset
            try:
                self.check_line(side, first, end); self.check_line(side, end, lower)
                return float(height), end
            except ValueError as exc:
                errors.append(str(exc))
        raise ValueError('No supported transport height: '+errors[-1])

    def plan_side(self, side, observation):
        grasp = self.grasp_goal(side, observation)
        self.motor.predict(side, grasp+[0.,0.,.055])
        offset = np.asarray(observation)-grasp
        errors = []
        for lift in self.routing['lift_candidates_m']:
            end = grasp+[0.,0.,lift]
            try:
                self.check_line(side, grasp, end)
                height, _ = self.transport(side, end, offset)
                return {'lift_m':float(lift),'transport_height_above_table_m':height}
            except ValueError as exc:
                errors.append(str(exc))
        raise ValueError('No supported lift/transport route: '+errors[-1])

    def choose_side(self, q):
        errors = []
        for side in ([self.forced_side] if self.forced_side else ('left','right')):
            try:
                self.route_details = self.plan_side(side, self.observation)
                self.side = side; self.offset = 0 if side == 'left' else 6
                return
            except ValueError as exc:
                errors.append(side+': '+str(exc))
        raise ValueError('; '.join(errors))

    def enter(self, stage, now, q):
        if self.routing.get('carry_visual_updates') and self.image_mode == 'live' and stage in ('align','lower'):
            age = now-self.last_observation_time
            if age > self.routing['maximum_carry_observation_age_s']:
                raise ValueError('No fresh RGB observation for the '+stage+' visual checkpoint.')
            current, _ = self.motor.geometry.pose(self.side,q[self.offset:self.offset+5])
            previous = self.grasp_offset.copy()
            self.grasp_offset = self.observation-current
            self.route_details.setdefault('visual_carry_updates',[]).append({
                'stage':stage,'time_s':float(now),'observation_age_s':float(age),
                'previous_offset_m':previous.tolist(),'observed_offset_m':self.grasp_offset.tolist(),
                'offset_change_mm':float(np.linalg.norm(self.grasp_offset-previous)*1000)})
        super().enter(stage, now, q)
        if stage == 'lift':
            self.fixed_end = self.pstart+[0.,0.,self.route_details['lift_m']]
        elif stage == 'align':
            height, self.fixed_end = self.transport(self.side, self.pstart, self.grasp_offset)
            self.route_details['actual_transport_height_above_table_m'] = height
        elif stage == 'retract':
            errors = []
            for distance in self.routing['retract_candidates_m']:
                end = self.pstart+[0.,0.,distance]
                try:
                    self.check_line(self.side,self.pstart,end)
                    self.fixed_end = end; self.route_details['retract_m'] = float(distance)
                    return
                except ValueError as exc:
                    errors.append(str(exc))
            raise ValueError('No supported retract route: '+errors[-1])


def plan_bottle_route(motor, observation, destination, table_z, image_mode, routing):
    """Choose one arm or a release/park/regrasp relay using neural route guards."""
    errors = []
    def check(side, start, goal):
        probe = RoutedRgbServoBottle(motor,goal,table_z,image_mode,routing,side)
        return probe.plan_side(side,np.asarray(start))
    for side in ('left','right'):
        try:
            detail = check(side,observation,destination)
            return {'kind':'direct','legs':[{'side':side,'destination':list(destination),'planned':detail}],
                    'prior_refusals':errors}
        except ValueError as exc:
            errors.append('direct '+side+': '+str(exc))
    shared_points = routing.get('shared_bottle_positions_m',[routing['shared_bottle_position_m']])
    for shared in shared_points:
        assumed_shared_grasp = np.r_[shared,table_z+.128]
        for first, second in (('left','right'),('right','left')):
            try:
                a = check(first,observation,shared)
                b = check(second,assumed_shared_grasp,destination)
                return {'kind':'table-supported relay','legs':[
                    {'side':first,'destination':list(shared),'planned':a},
                    {'side':second,'destination':list(destination),'planned':b}], 'prior_refusals':errors,
                    'shared_pose_note':'Nominal geometry only for preflight. The second leg must acquire its own fresh RGB observation.'}
            except ValueError as exc:
                errors.append('relay '+first+' to '+second+' via '+str(shared)+': '+str(exc))
    raise ValueError('; '.join(errors))
