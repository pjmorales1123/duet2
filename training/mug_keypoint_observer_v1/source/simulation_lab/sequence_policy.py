"""Recurrent neural action prediction from fitted RGB features and measured joints.

No demonstration actions, object poses, task stages or simulator clock at inference.
"""
import json
from pathlib import Path
import numpy as np
import torch
from .retrieval_policy import KEYS,image_vector,ObservationRejected

class SequenceNet(torch.nn.Module):
    def __init__(self,initial_context=False):
        super().__init__()
        self.initial_context=initial_context
        self.encoder=torch.nn.Sequential(torch.nn.Linear(112 if initial_context else 56,128),torch.nn.Tanh())
        self.gru=torch.nn.GRU(128,128,num_layers=2,batch_first=True)
        self.head=torch.nn.Sequential(torch.nn.Linear(128,128),torch.nn.Tanh(),torch.nn.Linear(128,240))
    def forward(self,x,hidden=None,context=None):
        if self.initial_context:
            if context is None:context=x[:,:1]
            x=torch.cat([x,context.expand(-1,x.shape[1],-1)],dim=-1)
        y,hidden=self.gru(self.encoder(x),hidden)
        return self.head(y).reshape(*x.shape[:2],20,12),hidden

class SequencePolicy(torch.nn.Module):
    def __init__(self,folder,device):
        super().__init__();folder=Path(folder)
        self.meta=json.loads((folder/'sequence.json').read_text())
        self.net=SequenceNet(self.meta['arguments'].get('initial_context',False)).to(device)
        from safetensors.torch import load_file
        self.net.load_state_dict(load_file(str(folder/'sequence.safetensors'),device=device));self.eval()
        with np.load(folder/'visual.npz',allow_pickle=False) as z:self.visual={k:z[k] for k in z.files}
        self.hidden=None;self.context=None;self.calls=0
        self.previous_endpoint=None;self.previous_chunk=None;self.tracking_waits=0;self.consecutive_waits=0
    def predict_action_chunk(self,batch):
        for key in ['observation.state',*KEYS]:
            if not torch.isfinite(batch[key]).all():raise ObservationRejected('Nonfinite sequence-policy observation.')
        if batch['observation.state'].shape!=(1,24) or any(batch[k].shape!=(1,3,240,320) for k in KEYS):
            raise ObservationRejected('Unexpected camera/joint observation dimensions.')
        state=batch['observation.state'][0].detach().cpu().numpy()
        images=[(batch[k][0].detach().cpu().permute(1,2,0).numpy()*255).round().clip(0,255).astype(np.uint8) for k in KEYS]
        a=self.visual;pixels=image_vector(images);projection=(pixels-a['mean'])@a['components'].T
        if np.mean((pixels-a['mean']-projection@a['components'])**2)>self.meta['reconstruction_limit']:
            raise ObservationRejected('Camera reconstruction outside fitted support.')
        features=np.r_[projection/a['scale'],(state-self.meta['state_mean'])/self.meta['state_std']]
        self.last_features=features.copy()
        x=torch.tensor(features,dtype=torch.float32,device=next(self.parameters()).device)[None,None]
        tolerance=self.meta.get('tracking_tolerance_rad')
        if tolerance and self.previous_endpoint is not None:
            error=float(np.max(np.abs(state[:5]-self.previous_endpoint[:5].detach().cpu().numpy())))
            if error>tolerance:
                self.tracking_waits+=1;self.consecutive_waits+=1
                if self.consecutive_waits>50:
                    raise ObservationRejected('Arm failed to reach the previous neural endpoint within ten seconds.')
                # Generic joint-feedback guard: wait for the already commanded
                # endpoint instead of advancing recurrent memory during a stall.
                if self.meta.get('tracking_replay_chunk'):
                    return self.previous_chunk
                return self.previous_endpoint[None,None].expand(1,20,-1)
        self.consecutive_waits=0
        if self.context is None:self.context=x.detach().clone()
        output,self.hidden=self.net(x,self.hidden,self.context);self.calls+=1
        if self.meta['arguments'].get('executed_only'):
            output=torch.cat([output[:,:,:5],output[:,:,4:5].expand(-1,-1,15,-1)],dim=2)
        decoded=output[:,0]*torch.tensor(self.meta['action_scale'],device=x.device)
        if self.meta.get('action_mode','relative')=='absolute':
            decoded=decoded+torch.tensor(self.meta['action_mean'],device=x.device)
        else:
            decoded=decoded+torch.tensor(state[:12],device=x.device)[None,None,:]
        self.previous_endpoint=decoded[0,4].detach().clone()
        self.previous_chunk=decoded.detach().clone()
        return decoded
    def details(self):
        return {'kind':'Two-layer GRU neural action-chunk policy with fitted PCA RGB features',
                'observation_calls':self.calls,'inference_demonstration_actions':False,
                'clock_or_teacher_phase_input':False,'hidden_state_resets':1,
                'tracking_guard_tolerance_rad':self.meta.get('tracking_tolerance_rad'),
                'tracking_guard_replays_last_predicted_chunk':self.meta.get('tracking_replay_chunk',False),
                'tracking_guard_wait_calls':self.tracking_waits}
