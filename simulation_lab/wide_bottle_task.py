"""Guarded wider-bottle task using the separately verified RGB/neural controller.

Motor inputs are camera estimates, requested destination and robot feedback.
The independent monitor reads privileged state exclusively to stop and score.
"""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

import mujoco
import numpy as np
from .autonomy import LiftReturn
from .bottle_refinement_runtime import RefinedBottleObserver
from .dinner_monitor import DinnerPhysicalMonitor
from .experiment_targets import with_bottle_destination
from .policy_control import apply_targets
from .rgb_servo_cameras import VIEWS, calibration, camera_argument
from .rgb_servo_openvino import OpenVinoCartesianMotorPolicy
from .rgb_servo_routing import RoutedRgbServoBottle, plan_bottle_route
from .scene import HOME

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROFILE = ROOT/'models/bottle_wide_v1/profile.json'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def artifact(name):
    path = (ROOT/name).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError('A wider-bottle dependency leaves the repository.')
    return path


def load_profile(path=None):
    profile = read(path or DEFAULT_PROFILE)
    if profile['schema'] != 'talos.bottle-wide-profile.v1':
        raise ValueError('Unknown wider-bottle profile.')
    for name, digest in profile['bound_files'].items():
        if hashlib.sha256(artifact(name).read_bytes()).hexdigest() != digest:
            raise ValueError('A verified wider-bottle dependency changed: '+name)
    audit = read(artifact(profile['physical_audit']))
    if not audit['gate_passed'] or not audit['all_118_outcomes_retained']:
        raise ValueError('Wider bottle control requires its complete physical gate.')
    if not all(all(v.values()) for v in audit['requirements'].values()):
        raise ValueError('Wider bottle physical requirements are incomplete.')
    return profile


def destination_for_plan(plan):
    """Use only existing typed language intents; never infer arbitrary targets."""
    steps = plan.get('steps', [])
    if len(steps) != 1 or steps[0]['kind'] != 'dinner_place' or steps[0]['object_id'] != 'bottle':
        raise ValueError('Wide bottle vision supports one bottle placement. Say “place the bottle” or “place the bottle at the right spot”.')
    step = steps[0]
    if step.get('arm', 'auto') != 'auto':
        raise ValueError('Wide bottle vision chooses the arm or table relay from the camera observation.')
    destination = step.get('destination') or {'kind': 'default'}
    if destination['kind'] == 'default':
        return [.10, -.115]
    if destination['kind'] == 'spot' and destination.get('side') == 'right' and not destination.get('far'):
        return [.20, .05]
    raise ValueError('Wide bottle vision has two tested destinations: its default place and the right spot.')


def wide_bottle_pose(seed, table_z):
    """Declared reset recipe; this function is never called during control."""
    rng = np.random.default_rng(seed)
    x, y, yaw = rng.uniform(-.14, .16), rng.uniform(-.18, -.06), rng.uniform(-.6, .6)
    return np.array([x, y, table_z+.001, math.cos(yaw/2), 0., 0., math.sin(yaw/2)])


class WideBottleTask(LiftReturn):
    def __init__(self, model, data, layout, destination, *, profile_path=None):
        super().__init__(model, data, layout)
        self.profile = load_profile(profile_path)
        if layout.get('scenario') != 'dinner':
            raise ValueError('Wide bottle vision requires the dinner scene.')
        if np.max(np.abs(data.qpos[:12]-np.array(HOME*2))) >= .035 or np.max(np.abs(data.qvel[:12])) >= .12:
            raise ValueError('Park both arms before starting wide bottle vision.')
        self.destination_xy = list(destination)
        if self.destination_xy not in self.profile['destinations_xy_m']:
            raise ValueError('Requested bottle destination has not passed the wider physical test.')
        self.observer = RefinedBottleObserver(artifact(self.profile['observer']))
        motor = artifact(self.profile['motor'])
        self.motor = OpenVinoCartesianMotorPolicy(motor/'openvino/motor.xml', motor/'motor.safetensors', model, device='CPU')
        self.routing = read(artifact(self.profile['routing']))
        self.camera_settings = read(artifact(self.profile['camera_protocol']))['camera_configurations'][self.profile['camera_configuration']]
        original = model.vis.quality.offsamples; model.vis.quality.offsamples = 0
        try:
            self.renderer = mujoco.Renderer(model, width=320, height=240)
        finally:
            model.vis.quality.offsamples = original
        self.option = mujoco.MjvOption(); self.option.geomgroup[3:] = 0
        self.route = self.controller = self.monitor = None
        self.leg_index = self.tick = 0
        self.epoch = self.waiting_since = 0.
        self.completed_legs, self.observation_log = [], []
        self.initial_others = {o['id']: data.body(o['id']).xpos.copy() for o in layout['objects'] if o['id'] != 'bottle'}
        self.maximum_other = 0.
        self.kind, self.stage, self.status = 'learned_bottle_wide', 'observe', 'running'
        self.object_id = self.skill = 'bottle'
        self.tube = {'id': 'bottle'}
        self.message = 'Locate the bottle in three RGB views and check a neural route.'
        self.final_physical = None

    def close(self):
        if getattr(self, 'renderer', None) is not None:
            self.renderer.close(); self.renderer = None

    def _finish(self, status, message, pause=True):
        super()._finish(status, message, pause)
        self.close()

    def apply_gripper_limit(self, targets):
        offset = 6 if self.controller and self.controller.side == 'right' else 0
        return apply_targets(self.model, self.data, targets, .25, offset)

    def _observe_rgb(self):
        images, calibrations = {}, {}
        for name in VIEWS:
            camera = camera_argument(name)
            camera.azimuth, camera.elevation, camera.distance = self.camera_settings[name]
            self.renderer.update_scene(self.data, camera=camera, scene_option=self.option)
            self.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
            images[name] = self.renderer.render().copy()
            calibrations[name] = calibration(self.renderer)
        return self.observer.observe(images, calibrations)

    def update(self, targets):
        if not self.active:
            return
        now = float(self.data.time)-self.started
        try:
            if self.tick % 20 == 0:
                estimate = self._observe_rgb()
                self.observation_log.append({'time_s': now, 'leg': self.leg_index, 'observation': estimate})
                if self.route is None and estimate['status'] == 'observed':
                    point = np.asarray(estimate['grasp_point_m'])
                    bounds = self.profile['source_workspace_m']
                    margin = self.profile['source_observation_margin_m']
                    if not (bounds['x'][0]-margin <= point[0] <= bounds['x'][1]+margin
                            and bounds['y'][0]-margin <= point[1] <= bounds['y'][1]+margin):
                        raise ValueError('The observed bottle is outside the tested wider source region.')
                    self.route = plan_bottle_route(self.motor, point, self.destination_xy, self.layout['table_z'], 'live', self.routing)
                if self.route and self.controller is None and estimate['status'] == 'observed':
                    leg = self.route['legs'][self.leg_index]
                    self.epoch = now
                    self.controller = RoutedRgbServoBottle(self.motor, leg['destination'], self.layout['table_z'], 'live', self.routing, leg['side'])
                    self.monitor = DinnerPhysicalMonitor(self.model, self.data, with_bottle_destination(self.layout, leg['destination']), 'bottle', leg['side'])
                if self.controller:
                    self.controller.accept_observation(estimate, now-self.epoch)
            if self.controller:
                targets[:] = self.controller.update(now-self.epoch, self.data.qpos[:12].copy(), self.data.qvel[:12].copy())
                self.stage = f'leg-{self.leg_index+1}:'+self.controller.stage
                self.side = self.controller.side
            # These privileged values are used only by the independent stop/score
            # checks, never by the RGB observer, route planner or neural motor.
            self.maximum_other = max(self.maximum_other, max((float(np.linalg.norm(self.data.body(n).xpos-p)) for n, p in self.initial_others.items()), default=0.))
            failure = self.monitor.update() if self.monitor else None
            if self.maximum_other > .004:
                failure = 'Cumulative non-target displacement exceeded 4 mm.'
            if self.monitor:
                self.metrics = dict(self.monitor.metrics)
                self.final_physical = self.monitor.report()
            if failure:
                raise ValueError(failure)
            if self.monitor and self.monitor.succeeded:
                self.completed_legs.append({'leg': self.leg_index, 'side': self.controller.side,
                    'destination': self.route['legs'][self.leg_index]['destination'], 'physical': self.monitor.report(),
                    'route_details': deepcopy(self.controller.route_details), 'stages': deepcopy(self.controller.history),
                    'accepted_rgb': self.controller.observations, 'rgb_refusals': self.controller.refusals})
                self.leg_index += 1
                if self.leg_index == len(self.route['legs']):
                    self._finish('succeeded', 'Bottle placed; every route leg physically released and both arms parked.')
                    return
                self.controller = self.monitor = None
                targets[:] = self.data.qpos[:12].copy()
                self.waiting_since = now
            elif self.controller and self.controller.status != 'running':
                raise ValueError(self.controller.message)
            elif self.route is None and now > 1.:
                raise ValueError('No confident initial RGB estimate.')
            elif self.route and self.controller is None and now-self.waiting_since > 2.:
                raise ValueError('No fresh RGB observation for the next relay leg.')
            if now > 100.:
                raise ValueError('Wide bottle control reached its physical time limit.')
            self.tick += 1
        except ValueError as exc:
            self._finish('failed', str(exc)+' Physics paused.')

    def snapshot(self):
        result = super().snapshot()
        total = len(self.route['legs']) if self.route else 1
        fraction = self.leg_index/total
        if self.controller:
            stages = list(self.controller.durations)
            if self.controller.stage in stages:
                fraction += stages.index(self.controller.stage)/max(1, len(stages))/total
        result.update(object_id='bottle', skill_id='bottle', policy_mode='learned_bottle_wide',
            visual_feedback_profile=self.profile['name'], visual_feedback_scope='bottle_approach_and_grasp',
            stage_label=self.stage.replace(':', ' · '), progress=1. if self.status == 'succeeded' else min(.99, fraction),
            observation='Three RGB views during approach/grasp; requested destination and motor feedback',
            physical_monitor='Privileged state for stop/score only; never motor target generation',
            destination_xy_m=self.destination_xy, route_kind=self.route['kind'] if self.route else None,
            completed_route_legs=len(self.completed_legs), total_route_legs=total, rgb_observations=len(self.observation_log),
            teacher_updates=0, inverse_solver_calls_during_control=0,
            cumulative_non_target_displacement_m=self.maximum_other,
            completion_note='Physical release and parked arms are required between relay legs.',
            results=[{'skill': 'bottle', 'status': self.status, 'message': self.message, 'metrics': dict(self.metrics)}],
            completed_steps=['bottle'] if self.status == 'succeeded' else [],
            stages=[{'id': 'observe', 'label': 'Locate bottle'}, {'id': 'manipulate', 'label': 'Grasp and place'},
                    {'id': 'verify', 'label': 'Release and park'}])
        return result


def make_sequence(model, data, layout, plan):
    """Prepare supported typed intents; full-sequence deployment is separately tested."""
    steps = plan.get('steps', [])
    if len(steps) != 1 or steps[0]['kind'] != 'set_table':
        return WideBottleTask(model, data, layout, destination_for_plan(plan))
    from .dinner_autonomy import SKILLS
    from .mug_visual_profile import load_profile as load_mug_profile
    from .mug_visual_control import VisualMugSequence
    _, mug_protocol = load_mug_profile()
    suite = artifact(mug_protocol['baseline_suite'])
    checkpoints = {name: suite.parent/path for name, path in read(suite).items()}

    class WideDinnerSequence(VisualMugSequence):
        def _next(self):
            skill = self.steps[len(self.results)]
            if skill == 'bottle':
                self.child = WideBottleTask(self.model, self.data, self.layout, [.10, -.115])
                self.stage = self.child.stage
            else:
                super()._next()

        def snapshot(self):
            result = super().snapshot()
            result.update(visual_feedback_profile='Wider bottle + live mug vision',
                          visual_feedback_scope='bottle_approach_and_grasp_then_late_mug_placement')
            return result

    sequence = WideDinnerSequence(model, data, layout, checkpoints, list(SKILLS), mug_protocol, 'live')
    sequence.plan = {'steps': list(SKILLS), 'reasons': ['Use RGB-guided bottle routing, then the existing dinner skills with late mug visual correction.'],
                     'interpreter': 'constrained_grammar', 'instruction': plan.get('instruction')}
    return sequence
