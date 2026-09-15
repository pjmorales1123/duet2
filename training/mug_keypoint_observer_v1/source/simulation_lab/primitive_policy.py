"""Camera-conditioned neural motion primitive with explicit joint-feedback timing.

Images condition the primitive at its start. Joint endpoint tracking gates progress;
this is not continuously vision-corrected manipulation or a recurrent network.
"""
import json
from pathlib import Path
import numpy as np
import torch
from .retrieval_policy import KEYS,image_vector,ObservationRejected

def primitive_image_vector(images,mode='raw'):
    vector=image_vector(images)
    if mode=='brightness_normalized':
        views=vector.reshape(3,-1)
        return (views/np.maximum(views.mean(axis=1,keepdims=True),.05)).ravel()
    if mode!='raw':raise ValueError('Unknown primitive image preprocessing.')
    return vector

class PrimitiveNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net=torch.nn.Sequential(torch.nn.Linear(45,256),torch.nn.SiLU(),torch.nn.Linear(256,256),torch.nn.SiLU(),torch.nn.Linear(256,256),torch.nn.SiLU(),torch.nn.Linear(256,12))
    def forward(self,visual,seconds):
        phase=seconds[...,None]/50
        angles=phase*torch.tensor([1,2,4,8,16,32],device=phase.device)*2*torch.pi
        return self.net(torch.cat([visual,phase,angles.sin(),angles.cos()],dim=-1))

class OpenVinoPrimitiveNet(torch.nn.Module):
    """Only the neural network runs in OpenVINO; preprocessing/guard stay explicit."""
    def __init__(self,folder,device):
        super().__init__()
        if device!='cpu':raise ValueError('OpenVINO adapter uses CPU tensors; select --device cpu.')
        import openvino as ov
        config=json.loads((folder/'openvino.json').read_text())
        self.register_parameter('_device_anchor',torch.nn.Parameter(torch.zeros(1),requires_grad=False))
        core=ov.Core();target=config['device']
        options={'PERFORMANCE_HINT':'LATENCY','INFERENCE_PRECISION_HINT':'f32'}
        if target=='CPU':options['INFERENCE_NUM_THREADS']=2
        self.compiled=core.compile_model(str(folder/'primitive.xml'),target,options)
        self.execution_devices=list(self.compiled.get_property('EXECUTION_DEVICES'))
    def forward(self,visual,seconds):
        values=self.compiled({'visual':visual.detach().cpu().numpy(),'seconds':seconds.detach().cpu().numpy()})[0]
        return torch.from_numpy(values.copy())

class PrimitivePolicy(torch.nn.Module):
    def __init__(self,folder,device):
        super().__init__();folder=Path(folder)
        self.meta=json.loads((folder/'primitive.json').read_text())
        if (folder/'openvino.json').exists():self.net=OpenVinoPrimitiveNet(folder,device)
        else:
            self.net=PrimitiveNet().to(device)
            from safetensors.torch import load_file
            self.net.load_state_dict(load_file(str(folder/'primitive.safetensors'),device=device))
        self.eval()
        with np.load(folder/'visual.npz',allow_pickle=False) as z:self.visual={k:z[k] for k in z.files}
        self.context=None;self.progress=0;self.previous=None;self.waits=0;self.total_waits=0
        self.feedback=self.meta.get('visual_feedback')
        self.visual_refreshes=0;self.initial_features=None
        if self.feedback and (self.feedback.get('mode')!='approach_refresh'
                or self.meta.get('visual_encoder')!='bottle_rgb_geometry'
                or not 0<float(self.feedback.get('until_progress_s',0))<=3.5
                or not 0<float(self.feedback.get('max_centroid_shift_px',0))<=12):
            raise ValueError('Unsupported visual feedback configuration.')
    def predict_action_chunk(self,batch):
        if batch['observation.state'].shape!=(1,24) or any(batch[k].shape!=(1,3,240,320) for k in KEYS):
            raise ObservationRejected('Unexpected camera/joint observation dimensions.')
        # CPU validation avoids waking a Torch reduction worker team from the
        # engine thread for every camera. Keep this check on a bounded CPU path.
        if any(not np.isfinite(batch[k].detach().cpu().numpy()).all() for k in ['observation.state',*KEYS]):
            raise ObservationRejected('Nonfinite primitive observation.')
        state=batch['observation.state'][0].detach().cpu().numpy()
        images=[(batch[k][0].detach().cpu().permute(1,2,0).numpy()*255).round().clip(0,255).astype(np.uint8) for k in KEYS]
        a=self.visual
        refresh=bool(self.feedback and self.progress/20<self.feedback['until_progress_s'])
        if self.meta.get('visual_encoder')=='bottle_rgb_geometry':
            from .bottle_vision import bottle_features
            # Initial-location conditioning cannot localize a bottle after the
            # hand occludes it. Later frames are shape/finite checked above.
            pixels=bottle_features(images) if self.context is None or refresh else a['mean'].copy()
            if self.initial_features is None:self.initial_features=pixels.copy()
            elif refresh and np.linalg.norm((pixels[:2]-self.initial_features[:2])*[320,240])>self.feedback['max_centroid_shift_px']:
                raise ObservationRejected('Bottle moved beyond the tested visual approach correction range.')
        elif self.meta.get('visual_encoder')=='dinner_rgb_geometry':
            from .dinner_vision import dinner_features
            pixels=dinner_features(images,self.meta['skill']) if self.context is None else a['mean'].copy()
        else:pixels=primitive_image_vector(images,self.meta.get('visual_preprocess','raw'))
        centered=pixels-a['mean']
        # These tiny matrix-vector products are slower when a desktop BLAS
        # launches many worker threads alongside rendering. Use bounded loops.
        projection=np.einsum('i,ji->j',centered,a['components'],optimize=False)
        reconstruction=np.einsum('i,ij->j',projection,a['components'],optimize=False)
        check_reconstruction=self.context is None or self.meta.get('reconstruction_check','every_query')=='every_query'
        support=self.meta.get('feature_support')
        if support and (self.context is None or refresh):
            lo,hi=np.asarray(support['min']),np.asarray(support['max'])
            if lo.shape!=pixels.shape or hi.shape!=pixels.shape or not np.isfinite([lo,hi]).all() or np.any(lo>hi):
                raise ObservationRejected('Invalid fitted camera support metadata.')
            if np.any(pixels<lo) or np.any(pixels>hi):raise ObservationRejected('Object observation lies outside the trained camera support.')
        if check_reconstruction and np.mean((centered-reconstruction)**2)>self.meta['reconstruction_limit']:
            raise ObservationRejected('Camera reconstruction outside fitted support.')
        device=next(self.parameters()).device
        if self.context is None or refresh:
            self.context=torch.tensor((projection/a['scale']-self.meta['visual_mean'])/self.meta['visual_std'],device=device,dtype=torch.float32)
            if refresh:self.visual_refreshes+=1
        offset=self.meta.get('arm_offset',0)
        if type(offset) is not int or offset not in (0,6):raise ObservationRejected('Invalid policy arm offset.')
        if self.previous is not None and np.max(np.abs(state[offset:offset+5]-self.previous[0,4,offset:offset+5].cpu().numpy()))>self.meta['tracking_tolerance_rad']:
            self.waits+=1;self.total_waits+=1
            if self.waits>50:raise ObservationRejected('Arm failed to track the neural primitive within ten seconds.')
            return self.previous
        self.waits=0
        seconds=(torch.arange(20,device=device)+self.progress)/20
        seconds=seconds.clamp(max=self.meta['max_seconds'])
        action=self.net(self.context[None].expand(20,-1),seconds)*torch.tensor(self.meta['action_std'],device=device)+torch.tensor(self.meta['action_mean'],device=device)
        self.previous=action[None].detach();self.progress+=4
        return self.previous
    def details(self):
        return {'kind':'Camera-conditioned neural motion primitive with programmed joint-feedback progress guard',
                'continuous_visual_correction':False,'initial_camera_conditioning':True,
                'pregrasp_visual_feedback':bool(self.feedback),
                'visual_feedback_scope':'RGB refresh during initial approach only; frozen before grasp' if self.feedback else 'initial image only',
                'visual_refresh_calls':self.visual_refreshes,
                'camera_support_check':self.meta.get('reconstruction_check','every_query'),
                'visual_encoder':self.meta.get('visual_encoder','camera_pca'),
                'neural_runtime':'OpenVINO' if hasattr(self.net,'compiled') else 'PyTorch',
                'neural_execution_devices':getattr(self.net,'execution_devices',[str(next(self.parameters()).device)]),
                'inference_demonstration_actions':False,'simulator_clock_or_object_pose_input':False,
                'internal_progress_endpoints':self.progress,'tracking_guard_wait_calls':self.total_waits}
