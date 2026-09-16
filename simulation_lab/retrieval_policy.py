"""Experimental observation-conditioned trajectory retrieval; not a neural ACT policy.

The temporal tie-breaker resolves indistinguishable stationary observations. It
never receives simulator time, task stage, object pose, contacts or episode name.
"""
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from .policy_observation import ObservationRejected

KEYS=['observation.images.'+c for c in ['overhead','left_wrist_cam','right_wrist_cam']]


def image_vector(images):
    return np.concatenate([np.asarray(Image.fromarray(rgb).resize((32,24),Image.Resampling.BOX),dtype=np.float32).ravel()/255 for rgb in images])


class RetrievalPolicy(torch.nn.Module):
    def __init__(self,folder):
        super().__init__()
        self.register_parameter('device_marker',torch.nn.Parameter(torch.zeros(1),requires_grad=False))
        with np.load(Path(folder)/'retrieval.npz',allow_pickle=False) as z:
            self.arrays={k:z[k] for k in z.files}
        self.meta=json.loads((Path(folder)/'retrieval.json').read_text())
        self.cursor=None;self.episode=None;self.trace=[];self.mixture=[]

    def reference(self,key,indices):
        if len(self.mixture)<2:return self.arrays[key][indices]
        a=self.arrays;relative=indices-a['bounds'][self.episode,0]
        return sum(weight*a[key][a['bounds'][episode,0]+relative] for episode,weight in self.mixture)

    def predict_action_chunk(self,batch):
        if batch['observation.state'].shape!=(1,24):raise ObservationRejected('Expected one 24-value joint observation.')
        for key in ['observation.state',*KEYS]:
            if not torch.isfinite(batch[key]).all():raise ObservationRejected('Nonfinite camera or joint observation.')
        if any(batch[key].shape!=(1,3,240,320) for key in KEYS):
            raise ObservationRejected('Expected three calibrated 320x240 RGB cameras.')
        state=batch['observation.state'][0].detach().cpu().numpy()
        images=[(batch[k][0].detach().cpu().permute(1,2,0).numpy()*255).round().clip(0,255).astype(np.uint8) for k in KEYS]
        a=self.arrays
        pixels=image_vector(images)
        projection=(pixels-a['mean'])@a['components'].T
        residual=float(np.mean((pixels-a['mean']-projection@a['components'])**2))
        if residual>self.meta.get('reconstruction_limit',float('inf')):
            raise ObservationRejected('Camera reconstruction error exceeds the fitted visual support.')
        visual=projection/a['scale']
        if self.cursor is None:
            candidates=np.concatenate([np.arange(s,min(s+5,e)) for s,e in a['bounds']])
            expected=None
        else:
            start,end=a['bounds'][self.episode]
            candidates=np.arange(max(start,self.cursor-2),min(end,self.cursor+13))
            expected=min(self.cursor+4,end-1)
        reference_state=self.reference('states',candidates)
        visual_distance=np.mean((self.reference('visual',candidates)-visual)**2,axis=1)
        qdist=np.mean(((reference_state[:,:12]-state[:12])/.03)**2,axis=1)
        vdist=np.mean(((reference_state[:,12:]-state[12:])/.15)**2,axis=1)
        score=visual_distance+qdist+vdist
        if expected is not None:score+=self.meta.get('progress_weight',1e-8)*(candidates-expected)**2
        best=int(np.argmin(score));index=int(candidates[best])
        if visual_distance[best]>9 or qdist[best]>25:
            raise ObservationRejected('Camera/joint observation is outside the retrieval support.')
        if self.episode is None:
            self.episode=next(i for i,(s,e) in enumerate(a['bounds']) if s<=index<e)
            self.mixture=[(self.episode,1.)]
            neighbors=int(self.meta.get('blend_neighbors',1))
            if neighbors>1 and score[best]>1e-9 and 'alignment' in a:
                # Blend only demonstrations whose offline phase timelines were identical.
                # No phase, position or episode identifier is supplied by the environment.
                relative=index-a['bounds'][self.episode,0]
                eligible=np.flatnonzero(a['alignment']==a['alignment'][self.episode])
                starts=a['bounds'][eligible,0]+relative
                distances=np.mean((a['visual'][starts]-visual)**2,axis=1)
                distances+=np.mean(((a['states'][starts,:12]-state[:12])/.03)**2,axis=1)
                distances+=np.mean(((a['states'][starts,12:]-state[12:])/.15)**2,axis=1)
                selected=np.argsort(distances)[:neighbors]
                weights=1/(distances[selected]+1e-8);weights/=weights.sum()
                self.mixture=[(int(eligible[i]),float(w)) for i,w in zip(selected,weights)]
        self.cursor=index
        self.trace.append({'index':index,'visual_distance':float(visual_distance[best]),'joint_distance':float(qdist[best]),'reconstruction_error':residual})
        end=a['bounds'][self.episode,1]
        indices=np.minimum(np.arange(index,index+20),end-1)
        return torch.from_numpy(self.reference('actions',indices).astype(np.float32))[None]

    def details(self):
        return {'kind':'PCA visual/proprioceptive retrieval with temporal tie-breaking',
                'progress_weight':self.meta.get('progress_weight',1e-8),
                'mixture':[{'demonstration':self.meta['episodes'][i],'weight':w} for i,w in self.mixture],
                'selected_demonstration':None if self.episode is None else self.meta['episodes'][self.episode],
                'trace':self.trace,'not_ACT':True}
