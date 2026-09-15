"""Neural dinner skills and explicit sequencing, with separate physical scoring.

Camera images and motor feedback are the policy inputs. Task selection is a
bounded language plan. No teacher update, action retrieval or IK at inference.
"""
from pathlib import Path
from copy import deepcopy
import time
import mujoco
import numpy as np
import torch
from .autonomy import LiftReturn
from .dinner_autonomy import SKILLS
from .dinner_monitor import DinnerPhysicalMonitor
from .policy_control import apply_targets
from .primitive_policy import PrimitivePolicy
from .retrieval_policy import KEYS, ObservationRejected
from .scene import HOME
from .policy_cameras import camera_argument
from .learned_skills import ALL_LEARNED_SKILLS,RELAY_SKILLS,SKILL_LABELS,object_for_skill


DEFAULT_SUITE=Path(__file__).resolve().parents[1]/'models'/'dinner_suite'/'suite.json'


def observe_scene(model,data):
    from .dinner_vision import scene_observation
    original=model.vis.quality.offsamples;model.vis.quality.offsamples=0
    try:renderer=mujoco.Renderer(model,height=240,width=320)
    finally:model.vis.quality.offsamples=original
    option=mujoco.MjvOption();option.geomgroup[3:]=0
    images=[]
    try:
        for name in ['overhead','table_left','table_right']:
            renderer.update_scene(data,camera=camera_argument(name),scene_option=option)
            renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW]=False
            images.append(renderer.render().copy())
    finally:renderer.close()
    return scene_observation(images)


class LearnedDinnerTask(LiftReturn):
    def __init__(self, model, data, layout, checkpoint, skill, *, policy_factory=PrimitivePolicy):
        super().__init__(model, data, layout)
        if layout.get('scenario') != 'dinner' or skill not in ALL_LEARNED_SKILLS:
            raise ValueError('Choose a supported dinner skill in the dinner scene.')
        if np.max(np.abs(data.qpos[:12]-np.array(HOME*2))) > .035:
            raise ValueError('Both arms must be parked before starting a learned skill.')
        self.policy = policy_factory(checkpoint, 'cpu')
        self.object_id=object_for_skill(skill)
        if self.policy.meta.get('skill','bottle') != self.object_id:
            raise ValueError('Checkpoint does not match the requested skill.')
        if skill in RELAY_SKILLS and self.policy.meta.get('logical_skill')!=skill:
            raise ValueError('Relay checkpoint does not match its trained leg.')
        self.skill = skill
        self.offset = self.policy.meta.get('arm_offset', 0)
        self.side = 'left' if self.offset == 0 else 'right'
        if skill in RELAY_SKILLS and not skill.endswith(self.side):raise ValueError('Relay arm does not match its checkpoint.')
        self.cap = self.policy.meta.get('gripper_cap_nm', .25)
        self.inactive_slice=slice(6,12) if self.offset==0 else slice(0,6)
        hold_mode=self.policy.meta.get('inactive_arm_control')
        if hold_mode not in (None,'hold_previous_targets'):raise ValueError('Unknown inactive-arm control contract.')
        self.inactive_targets=data.ctrl[self.inactive_slice].copy() if hold_mode else None
        monitor_layout=layout
        if skill in RELAY_SKILLS:
            destination=np.asarray(self.policy.meta.get('trained_destination_m'),dtype=float)
            if destination.shape!=(3,) or not np.isfinite(destination).all():raise ValueError('Relay checkpoint requires a fixed trained destination.')
            monitor_layout=deepcopy(layout)
            monitor_layout['targets']=[t for t in layout['targets'] if t['object_id']!=self.object_id]
            monitor_layout['targets'].append({'id':skill+'_destination','object_id':self.object_id,'position_m':destination.tolist()})
        self.monitor = DinnerPhysicalMonitor(model, data, monitor_layout, self.object_id, self.side)
        self.tube = self.monitor.observer.tube
        self.kind, self.stage, self.status = 'learned_dinner', 'neural_control', 'running'
        self.message = 'Neural '+SKILL_LABELS[skill]+' control from initial camera images and motor feedback.'
        self.tick = 0
        self.chunk = None
        self.inference_ms = []
        self.option = mujoco.MjvOption(); self.option.geomgroup[3:] = 0
        original = model.vis.quality.offsamples
        model.vis.quality.offsamples = 0
        try: self.renderer = mujoco.Renderer(model, height=240, width=320)
        finally: model.vis.quality.offsamples = original

    def close(self):
        if getattr(self, 'renderer', None) is not None:
            self.renderer.close(); self.renderer = None

    def _finish(self, status, message, pause=True):
        super()._finish(status, message, pause)
        self.close()

    def apply_gripper_limit(self, targets):
        if self.inactive_targets is not None:
            targets=np.asarray(targets).copy()
            targets[self.inactive_slice]=self.inactive_targets
        return apply_targets(self.model, self.data, targets, self.cap, self.offset)

    def _observation(self):
        batch = {'observation.state': torch.from_numpy(np.r_[self.data.qpos[:12], self.data.qvel[:12]].astype('float32'))[None]}
        names=self.policy.meta.get('cameras',[key.removeprefix('observation.images.') for key in KEYS])
        if len(names)!=3:raise ObservationRejected('Expected three policy camera views.')
        for key,name in zip(KEYS,names):
            self.renderer.update_scene(self.data, camera=camera_argument(name), scene_option=self.option)
            self.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
            # Keep image conversion off Torch's CPU worker pool in the UI thread.
            rgb = self.renderer.render().astype('float32')/255
            batch[key] = torch.from_numpy(rgb.transpose(2, 0, 1).copy())[None]
        return batch

    def update(self, targets):
        if not self.active: return
        failure = self.monitor.update()
        self.metrics = dict(self.monitor.metrics)
        if failure:
            targets[:] = self.data.qpos[:12]
            self._finish('failed', failure+' Physics paused.')
            return
        if self.monitor.succeeded:
            targets[:] = self.data.qpos[:12]
            self._finish('succeeded', 'Neural '+SKILL_LABELS[self.skill]+' completed with physical release and both arms parked.')
            return
        if self.data.time-self.started > self.policy.meta['max_seconds']+35:
            targets[:] = self.data.qpos[:12]
            self._finish('failed', 'Neural skill did not reach its physical goal before timeout.')
            return
        try:
            if self.tick % 40 == 0:
                observation = self._observation()
                started = time.perf_counter()
                with torch.inference_mode(): chunk = self.policy.predict_action_chunk(observation)[0].numpy()
                self.inference_ms.append((time.perf_counter()-started)*1000)
                if chunk.shape != (20, 12) or not np.isfinite(chunk).all():
                    raise ObservationRejected('Invalid neural action chunk.')
                self.chunk = np.clip(chunk, self.model.actuator_ctrlrange[:, 0], self.model.actuator_ctrlrange[:, 1])
            u = (self.tick % 40)/10; i = int(u)
            targets[:] = self.chunk[i]*(1-(u-i))+self.chunk[i+1]*(u-i)
            if self.inactive_targets is not None:targets[self.inactive_slice]=self.inactive_targets
            self.tick += 1
        except ObservationRejected as exc:
            targets[:] = self.data.qpos[:12]
            self._finish('failed', str(exc)+' Physics paused.')

    def snapshot(self):
        result = super().snapshot()
        result.update(object_id=self.object_id, skill_id=self.skill, skill_label=SKILL_LABELS[self.skill], arm=self.side, policy_mode='learned_dinner',
                      observation='initial three RGB cameras and motor-feedback progress guard',
                      physical_monitor='privileged state for stop/score only; never motor target generation',
                      stage_label='Neural '+SKILL_LABELS[self.skill]+' movement', teacher_updates=0,
                      policy_details=self.policy.details(),
                      inference_median_ms=float(np.median(self.inference_ms)) if self.inference_ms else None,
                      completion_note='Physical release and parked arms required before the next skill.')
        result['inactive_arm_control']=self.policy.meta.get('inactive_arm_control','neural targets')
        result['stages'] = [{'id': 'neural_control', 'label': 'Camera-conditioned neural movement'}]
        result['progress'] = 1. if self.status == 'succeeded' else min(.99, self.policy.progress/20/self.policy.meta['max_seconds'])
        return result


class LearnedDinnerSequence(LiftReturn):
    def __init__(self, model, data, layout, checkpoints, skills, *, policy_factory=PrimitivePolicy):
        super().__init__(model, data, layout)
        if not skills or any(s not in ALL_LEARNED_SKILLS for s in skills): raise ValueError('Invalid learned dinner sequence.')
        if len(skills) != len(set(skills)): raise ValueError('Repeated placement needs a separately trained destination.')
        if any(s not in checkpoints for s in skills):raise ValueError('The suite does not contain every requested learned skill.')
        self.checkpoints = {s: Path(checkpoints[s]) for s in skills}
        self.policy_factory = policy_factory
        if any(not (p/'primitive.json').is_file() for p in self.checkpoints.values()):
            raise ValueError('A requested learned skill has no checkpoint; no programmed fallback is used.')
        self.steps, self.results = list(skills), []
        self.child = None
        self.kind, self.status = 'learned_dinner_sequence', 'running'
        self.placed = {}
        self._next()

    def _next(self):
        skill = self.steps[len(self.results)]
        self.child = LearnedDinnerTask(self.model, self.data, self.layout, self.checkpoints[skill], skill, policy_factory=self.policy_factory)
        self.stage = self.child.stage

    def close(self):
        if self.child: self.child.close()

    def cancel(self, targets):
        if self.active:
            self.child.cancel(targets)
            self._finish('cancelled', 'Sequence cancelled; physics paused.', True)

    def apply_gripper_limit(self, targets):
        return self.child.apply_gripper_limit(targets)

    def update(self, targets):
        if not self.active: return
        self.child.update(targets)
        self.stage, self.message, self.side, self.tube = self.child.stage, self.child.message, self.child.side, self.child.tube
        if self.child.active: return
        self.results.append({'skill': self.child.skill, **self.child.snapshot()})
        if self.child.status != 'succeeded':
            self._finish(self.child.status, self.child.message, True)
            return
        if self.child.object_id != 'drawer':
            self.placed[self.child.object_id] = self.data.body(self.child.object_id).xpos.copy()
        if len(self.results) < len(self.steps):
            try: self._next()
            except ValueError as exc: self._finish('failed', str(exc), True)
            return
        disturbed = max((float(np.linalg.norm(self.data.body(n).xpos-p)) for n, p in self.placed.items()), default=0.)
        if 'drawer' in self.steps and self.data.joint('drawer_slide').qpos[0]<.105:
            self._finish('failed','A later skill closed the drawer.',True)
        elif disturbed > .004:
            self._finish('failed', 'A later skill disturbed an earlier placement.', True)
        else:
            self._finish('succeeded', 'Completed all requested neural dinner skills; both arms parked.', True)

    def snapshot(self):
        result = self.child.snapshot() if self.child else super().snapshot()
        skill_elapsed=result.get('elapsed_s',0.)
        result.update(status=self.status, active=self.active, kind=self.kind, message=self.message,
                      steps=self.steps, completed_steps=[r['skill'] for r in self.results if r['status'] == 'succeeded'],
                      results=list(self.results), policy_mode='learned_dinner', recording_id=self.recording_id)
        result['elapsed_s']=round(float(self.data.time)-self.started,2) if self.active else getattr(self,'elapsed',0.)
        result['skill_elapsed_s']=skill_elapsed
        completed=sum(r['status']=='succeeded' for r in self.results)
        result['progress'] = 1. if self.status == 'succeeded' else min(.99, (completed+result['progress'])/len(self.steps))
        if hasattr(self,'plan'):result['intent_plan']=self.plan
        return result
