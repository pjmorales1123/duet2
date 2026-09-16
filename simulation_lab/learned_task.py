"""Live camera/joint-driven bottle policy with a separate physical success monitor.

The monitor can stop execution, but never generates joint targets. No teacher
updates, action retrieval, attachments, or state restoration occur in this task.
"""
import time
from pathlib import Path
import mujoco,numpy as np,torch
from .autonomy import LiftReturn
from .dinner_autonomy import DinnerTask
from .primitive_policy import PrimitivePolicy
from .retrieval_policy import KEYS,ObservationRejected
from .policy_control import apply_targets
from .scene import HOME

DEFAULT_CHECKPOINT=Path(__file__).resolve().parents[1]/'models'/'bottle_visual'
LEGACY_CHECKPOINT=Path(__file__).resolve().parents[1]/'models'/'bottle_primitive'


class LearnedBottleTask(LiftReturn):
    def __init__(self,model,data,layout,checkpoint=DEFAULT_CHECKPOINT):
        super().__init__(model,data,layout)
        if layout.get('scenario')!='dinner':raise ValueError('The learned policy requires the dinner scene.')
        if np.max(np.abs(data.qpos[:12]-np.array(HOME*2)))>.025:
            raise ValueError('Reset the scene first: the learned bottle skill starts with parked arms.')
        if not (Path(checkpoint)/'primitive.json').is_file():
            raise ValueError('Learned checkpoint missing. Install the packaged bottle model first.')
        torch.set_num_threads(2)
        self.policy=PrimitivePolicy(checkpoint,'cpu')
        self.renderer=None
        self.observer=DinnerTask(model,data,layout)
        self.tube=next(o for o in layout['objects'] if o['id']=='bottle');self.side='left'
        self.observer._select_item('left',self.tube)
        self.observer.grip_torque=.25
        self.parked=data.qpos[6:12].copy();self.destination_position=self.observer.destination_position.copy()
        self.kind='learned_bottle';self.stage='neural_control';self.status='running'
        self.message='Neural bottle control from camera images and motor feedback.'
        self.tick=0;self.chunk=None;self.held=self.best_hold=self.stable=self.bad_gap=self.longest_gap=0.
        self.max_other=self.max_parked=0.;self.inference_ms=[];self.last_observed_time=float(data.time)
        self.policy_details=self.policy.details()
        self.option=mujoco.MjvOption();self.option.geomgroup[3:]=0
        # Match training anti-aliasing at context creation, then restore the
        # visual-only setting. No integration state or dynamics is changed.
        self.original_offsamples=model.vis.quality.offsamples
        model.vis.quality.offsamples=0
        try:self.renderer=mujoco.Renderer(model,height=240,width=320)
        finally:model.vis.quality.offsamples=self.original_offsamples

    def close(self):
        if self.renderer is not None:self.renderer.close();self.renderer=None

    def _finish(self,status,message,pause=False):
        super()._finish(status,message,pause);self.close()

    def apply_gripper_limit(self,targets):
        return apply_targets(self.model,self.data,targets,.25)

    def _observation(self):
        raw={'observation.state':torch.tensor(np.r_[self.data.qpos[:12],self.data.qvel[:12]],dtype=torch.float32)[None]}
        for key in KEYS:
            self.renderer.update_scene(self.data,camera=key.removeprefix('observation.images.'),scene_option=self.option)
            self.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW]=False
            rgb=self.renderer.render().copy()
            raw[key]=torch.from_numpy(rgb).permute(2,0,1)[None].float()/255
        return raw

    def _monitor(self):
        dt=float(self.data.time)-self.last_observed_time;self.last_observed_time=float(self.data.time)
        forces,lift,up,both=self.observer._observe()
        collision=self.observer._collision(self.data,allow_tube=True,penetration=.0008)
        if collision:self.observer.metrics['unexpected_collisions']+=1
        self.held=self.held+dt if both and lift>.05 and forces['external']<.02 else 0.
        self.best_hold=max(self.best_hold,self.held)
        self.bad_gap=self.bad_gap+dt if lift>.012 and not both and forces['base']<.02 else 0.
        self.longest_gap=max(self.longest_gap,self.bad_gap)
        self.max_other=max(self.max_other,max(float(np.linalg.norm(self.data.body(n).xpos-p)) for n,p in self.observer.others.items()))
        self.max_parked=max(self.max_parked,float(np.rad2deg(np.max(np.abs(self.data.qpos[6:12]-self.parked)))))
        error=float(np.linalg.norm(self.data.body('bottle').xpos[:2]-self.destination_position[:2]))
        dof=self.model.joint('bottle_free').dofadr[0];speed=np.linalg.norm(self.data.qvel[dof:dof+3])
        placed=error<.012 and up>np.cos(np.deg2rad(5)) and speed<.003 and forces['base']>.02 and max(forces['fixed'],forces['moving'])<.02
        self.stable=self.stable+dt if placed else 0.
        self.metrics=dict(self.observer.metrics,gripper_torque_limit_nm=.25,placement_error_mm=error*1000,hold_verified_s=self.best_hold,
                          stable_release_s=self.stable,longest_unsupported_gap_s=self.longest_gap,
                          other_object_max_displacement_m=self.max_other,parked_arm_max_motion_deg=self.max_parked)
        if collision:
            self._finish('failed','Physical monitor stopped the policy: unexpected arm contact '+collision+'.',True)
        elif self.max_other>.004 or self.max_parked>1 or self.longest_gap>.18:
            self._finish('failed','Physical monitor stopped the policy: lost support or disturbed another object/arm.',True)
        elif self.best_hold>=1.49 and self.stable>=1.:
            self._finish('succeeded','Neural policy physically lifted, held, released and placed the bottle.',True)
        elif self.data.time-self.started>60:
            self._finish('failed','The learned policy did not complete the physical goal within 60 simulated seconds.',True)

    def update(self,targets):
        if not self.active:return
        self._monitor()
        if not self.active:targets[:]=self.data.qpos[:12];return
        try:
            if self.tick%40==0:
                observation=self._observation();started=time.perf_counter()
                with torch.inference_mode():chunk=self.policy.predict_action_chunk(observation)[0].cpu().numpy()
                self.inference_ms.append((time.perf_counter()-started)*1000)
                if chunk.shape!=(20,12) or not np.isfinite(chunk).all():raise ObservationRejected('Invalid learned action chunk.')
                self.chunk=np.clip(chunk,self.model.actuator_ctrlrange[:12,0],self.model.actuator_ctrlrange[:12,1])
                self.policy_details=self.policy.details()
            u=(self.tick%40)/10;i=int(u)
            targets[:]=self.chunk[i]*(1-(u-i))+self.chunk[i+1]*(u-i);self.tick+=1
        except ObservationRejected as exc:
            targets[:]=self.data.qpos[:12];self._finish('failed',str(exc)+' Physics paused.',True)

    def snapshot(self):
        result=super().snapshot();result.update(object_id='bottle',stage_label='Neural camera-conditioned motion',
            observation=('initial overhead RGB bottle localization and motor feedback' if self.policy.meta.get('visual_encoder')=='bottle_rgb_geometry' else 'three initial RGB cameras and motor feedback'),policy_mode='learned_bottle',
            physical_monitor='privileged simulator state used only for success/failure stops, never action generation',
            policy_details=self.policy_details,teacher_updates=0,
            inference_median_ms=float(np.median(self.inference_ms)) if self.inference_ms else None,
            timing='200 Hz physics, 20 Hz action endpoints, 5 Hz neural queries. Synchronous inference can slow wall time; display rendering is independent.')
        result['progress']=1. if self.status=='succeeded' else min(.99,self.tick*.005/45)
        # Do not expose the inherited chemistry teacher's stage list as though
        # the neural policy followed those programmed stages.
        result['stages']=[{'id':'neural_control','label':'Camera-conditioned neural movement'}]
        result['policy_mode']='learned_bottle' if self.policy.meta.get('visual_encoder')=='bottle_rgb_geometry' else 'learned_bottle_legacy'
        result['completion_note']='Success verifies the released bottle; reset before another learned trial because the working arm may not yet be parked.'
        return result
