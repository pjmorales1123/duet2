"""Experimental late-placement RGB corrections around the unchanged mug policy.

The production dinner classes are inherited unchanged. Only this experimental
mug policy adds current-image corrections; privileged monitoring remains separate.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time
import mujoco
import numpy as np
import openvino as ov
import torch
from .learned_dinner import LearnedDinnerTask,LearnedDinnerSequence
from .primitive_policy import PrimitivePolicy
from .retrieval_policy import ObservationRejected
from .mug_keypoint_observer import decoded_outputs,reconstruct
from .mug_correction_runtime import LocalMugMotor
from .mug_visual_geometry import correction_request
from .rgb_servo_cameras import calibration

ROOT=Path(__file__).resolve().parents[1]


class MugStereoObserver:
    def __init__(self,folder):
        folder=Path(folder)
        self.protocol=json.loads((folder/'protocol.json').read_text())
        manifest=json.loads((folder/'manifest.json').read_text())
        for name in ('openvino/observer.xml','openvino/observer.bin'):
            if hashlib.sha256((folder/name).read_bytes()).hexdigest()!=manifest['files'][name]:
                raise ValueError('Mug observer IR changed.')
        core=ov.Core()
        self.compiled=core.compile_model(str(folder/'openvino/observer.xml'),'CPU',
            {'PERFORMANCE_HINT':'LATENCY','INFERENCE_PRECISION_HINT':'f32','INFERENCE_NUM_THREADS':2})

    def observe(self,images,calibrations):
        images=np.asarray(images)
        if images.shape!=(2,240,320,3) or images.dtype!=np.uint8:
            raise ValueError('Two uint8 stereo RGB images are required.')
        start=time.perf_counter()
        result=self.compiled([images.transpose(0,3,1,2).astype(np.float32)/255])
        with torch.inference_mode():
            decoded={key:value.numpy() for key,value in decoded_outputs(tuple(torch.from_numpy(result[i].copy()) for i in range(3))).items()}
        observation=reconstruct(decoded,calibrations,self.protocol['inference'])
        observation['observer_wall_ms']=(time.perf_counter()-start)*1000
        return observation


class MugCorrectionPolicy(PrimitivePolicy):
    def __init__(self,folder,device,robot_model,protocol,mode,*,observer=None,motor=None):
        super().__init__(folder,device)
        if mode not in ('live','frozen') or self.meta.get('skill')!='mug':
            raise ValueError('The experimental policy supports only live/frozen mug correction.')
        self.experiment=deepcopy(protocol)
        self.settings=self.experiment['control']
        self.mode=mode
        self.observer=observer if observer is not None else MugStereoObserver(ROOT/protocol['observer'])
        self.motor=motor if motor is not None else LocalMugMotor(ROOT/protocol['motor'],robot_model)
        self.ranges=np.asarray(robot_model.actuator_ctrlrange).copy()
        self.joint_offset=np.zeros(5)
        self.frozen_images=None;self.frozen_calibrations=None
        self.corrections=[]

    def _adjusted(self,nominal,phase,offset):
        seconds=phase+np.arange(20)/20
        start=self.settings['keep_offset_until_nominal_seconds']
        end=self.settings['taper_offset_end_nominal_seconds']
        fraction=np.clip((seconds-start)/(end-start),0,1)
        weights=1-fraction**3*(10+fraction*(-15+6*fraction))
        adjusted=nominal.clone()
        change=torch.tensor(weights[:,None]*offset[None],dtype=nominal.dtype,device=nominal.device)
        adjusted[0,:,6:11]+=change
        active=adjusted[0,:,6:11].detach().cpu().numpy()
        if not np.isfinite(active).all() or np.any(active<self.ranges[6:11,0]) or np.any(active>self.ranges[6:11,1]):
            raise ObservationRejected('Corrected active targets exceed actuator limits.')
        return adjusted

    def predict_action_chunk(self,batch):
        before=self.progress
        nominal=super().predict_action_chunk(batch)
        if self.progress==before:
            # The base guard returns the previously issued corrected chunk.
            # Waiting cannot add the same offset twice or accumulate correction.
            return nominal
        phase=before/20
        proposed=self.joint_offset.copy()
        active=self.settings['correction_nominal_seconds'][0]-1e-9<=phase<=self.settings['correction_nominal_seconds'][1]+1e-9
        record=None
        if active:
            record={'nominal_seconds':phase,'mode':self.mode,'status':'pending'}
            try:
                images=np.asarray(batch['mug.rgb'])
                calibrations=batch['mug.calibrations']
                if self.mode=='frozen' and self.frozen_images is None:
                    self.frozen_images=images.copy();self.frozen_calibrations=deepcopy(calibrations)
                used=self.frozen_images if self.mode=='frozen' else images
                used_calibrations=self.frozen_calibrations if self.mode=='frozen' else calibrations
                record['current_rgb_sha256']=[hashlib.sha256(image.tobytes()).hexdigest() for image in images]
                record['used_rgb_sha256']=[hashlib.sha256(image.tobytes()).hexdigest() for image in used]
                observation=self.observer.observe(used,used_calibrations)
                record['observation']=observation
                origin,request=correction_request(observation,self.settings)
                joints=batch['observation.state'][0,6:11].detach().cpu().numpy()
                motor=self.motor.predict(joints,request)
                record.update(estimated_origin_m=origin.tolist(),requested_translation_m=request.tolist(),motor=motor)
                if not motor['accepted']:
                    raise ObservationRejected('The local neural motor refused the visual correction.')
                proposed+=np.asarray(motor['joint_delta_rad'])
                if np.max(abs(proposed))>self.settings['maximum_cumulative_joint_offset_rad']:
                    raise ObservationRejected('Cumulative visual correction exceeds the declared joint range.')
            except (ValueError,KeyError,ObservationRejected) as exc:
                record.update(status='refused',reason=str(exc));self.corrections.append(record)
                raise ObservationRejected(str(exc)) from exc
        if not np.any(proposed):
            self.joint_offset=proposed
            if record is not None:
                record.update(status='issued',joint_offset_rad=self.joint_offset.tolist())
                self.corrections.append(record)
            return nominal
        try:
            corrected=self._adjusted(nominal,phase,proposed)
        except ObservationRejected as exc:
            if record is not None:
                record.update(status='refused',reason=str(exc));self.corrections.append(record)
            raise
        self.joint_offset=proposed
        # The next base tracking check must compare against actual issued targets.
        self.previous=corrected.detach()
        if record is not None:
            record.update(status='issued',joint_offset_rad=self.joint_offset.tolist())
            self.corrections.append(record)
        return self.previous

    def details(self):
        result=super().details()
        result.update(continuous_visual_correction=self.mode=='live',
            visual_feedback_scope='Late mug placement only; '+('current stereo RGB' if self.mode=='live' else 'frozen initial correction images'),
            local_motor='OpenVINO CPU neural differential map with explicit joint limiting and forward-only refusal',
            correction_queries=len(self.corrections),correction_mode=self.mode,
            correction_seconds=self.settings['correction_nominal_seconds'],
            inference_object_state=False,inference_inverse_kinematics=False)
        return result


class VisualMugTask(LearnedDinnerTask):
    def __init__(self,model,data,layout,checkpoint,protocol,mode):
        factory=lambda folder,device:MugCorrectionPolicy(folder,device,model,protocol,mode)
        super().__init__(model,data,layout,checkpoint,'mug',policy_factory=factory)
        observer=self.policy.observer.protocol
        camera_file=ROOT/observer['camera_protocol']
        if hashlib.sha256(camera_file.read_bytes()).hexdigest()!=observer['camera_protocol_sha256']:
            raise ValueError('Frozen observer camera settings changed.')
        camera_protocol=json.loads(camera_file.read_text())
        self.stereo_settings=[camera_protocol['camera_configurations'][observer['camera_configuration']][slot] for slot in observer['camera_slots']]

    def _observation(self):
        batch=super()._observation()
        phase=self.policy.progress/20
        low,high=self.policy.settings['correction_nominal_seconds']
        if low-.2-1e-9<=phase<=high+.4+1e-9:
            images,cameras=[],[]
            for settings in self.stereo_settings:
                camera=mujoco.MjvCamera();camera.type=mujoco.mjtCamera.mjCAMERA_FREE
                camera.lookat[:]=settings['lookat']
                for name in ('distance','azimuth','elevation'):setattr(camera,name,settings[name])
                self.renderer.update_scene(self.data,camera=camera,scene_option=self.option)
                self.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW]=False
                images.append(self.renderer.render().copy());cameras.append(calibration(self.renderer))
            batch['mug.rgb']=np.stack(images);batch['mug.calibrations']=cameras
        return batch

    def snapshot(self):
        result=super().snapshot()
        result.update(observation='Initial RGB neural trajectory plus '+self.policy.mode+' stereo late-placement correction',
            mug_visual_corrections=list(self.policy.corrections),mug_correction_mode=self.policy.mode)
        return result


class VisualMugSequence(LearnedDinnerSequence):
    def __init__(self,model,data,layout,checkpoints,skills,protocol,mode):
        self.visual_protocol=protocol;self.visual_mode=mode
        super().__init__(model,data,layout,checkpoints,skills)

    def _next(self):
        skill=self.steps[len(self.results)]
        if skill=='mug':
            self.child=VisualMugTask(self.model,self.data,self.layout,self.checkpoints[skill],self.visual_protocol,self.visual_mode)
            self.stage=self.child.stage
        else:
            super()._next()
