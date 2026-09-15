"""Shared, explicit normalization for ACT training and camera-only rollout."""
import json
from pathlib import Path
import numpy as np
import torch
from lerobot.configs.types import FeatureType,PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy

CAMERAS=['overhead','left_wrist_cam','right_wrist_cam']
IMAGE_KEYS=['observation.images.'+c for c in CAMERAS]

def episode_indices(lineage, name):
    offset=0
    for row in lineage['episodes']:
        if row['source']==name:return list(range(offset,offset+row['frames']))
        offset+=row['frames']
    raise ValueError('Episode absent from dataset lineage: '+name)

def learning_loss(policy,batch,objective='vae'):
    # eval() changes VAE/dropout behavior but does not disable gradients.
    # ACT's eval forward uses the same zero latent as predict_action_chunk.
    policy.train(objective=='vae')
    return policy(batch)

def normalization(stats):
    result={}
    for key,floor in [('observation.state',.02),('action',.02)]:
        result[key]={'mean':np.asarray(stats[key]['mean']).tolist(),
                     'std':np.maximum(np.asarray(stats[key]['std']),floor).tolist()}
    result['images']={'mean':[.485,.456,.406],'std':[.229,.224,.225]}
    return result

def normalize(batch,stats,device):
    result={}
    for key in ['observation.state','action']:
        if key in batch:
            values=torch.as_tensor(batch[key],device=device,dtype=torch.float32)
            if key=='action' and stats.get('action_representation')=='relative':
                values=values-torch.as_tensor(batch['observation.state'],device=device,dtype=torch.float32)[:,None,:12]
            result[key]=(values-torch.tensor(stats[key]['mean'],device=device))/torch.tensor(stats[key]['std'],device=device)
    for key in IMAGE_KEYS:
        image=torch.as_tensor(batch[key],device=device,dtype=torch.float32)
        result[key]=(image-torch.tensor(stats['images']['mean'],device=device)[None,:,None,None])/torch.tensor(stats['images']['std'],device=device)[None,:,None,None]
    if 'action_is_pad' in batch:result['action_is_pad']=batch['action_is_pad'].to(device)
    return result

def denormalize(action,stats,state=None):
    result=action*torch.as_tensor(stats['action']['std'],device=action.device)+torch.as_tensor(stats['action']['mean'],device=action.device)
    if stats.get('action_representation')=='relative':
        if state is None:raise ValueError('Relative actions require the measured observation state.')
        result=result+torch.as_tensor(state,device=action.device,dtype=action.dtype)[:,None,:12]
    return result

def config(device='cpu',pretrained=True):
    return ACTConfig(device=device,chunk_size=20,n_action_steps=4,
        input_features={'observation.state':PolicyFeature(type=FeatureType.STATE,shape=(24,)),
                        **{key:PolicyFeature(type=FeatureType.VISUAL,shape=(3,240,320)) for key in IMAGE_KEYS}},
        output_features={'action':PolicyFeature(type=FeatureType.ACTION,shape=(12,))},
        pretrained_backbone_weights='ResNet18_Weights.IMAGENET1K_V1' if pretrained else None)

def load_policy(folder,device):
    folder=Path(folder)
    if (folder/'primitive.safetensors').exists():
        from simulation_lab.primitive_policy import PrimitivePolicy
        return PrimitivePolicy(folder,device),json.loads((folder/'talos_normalization.json').read_text())
    if (folder/'sequence.safetensors').exists():
        from simulation_lab.sequence_policy import SequencePolicy
        return SequencePolicy(folder,device),json.loads((folder/'talos_normalization.json').read_text())
    if (folder/'retrieval.npz').exists():
        from simulation_lab.retrieval_policy import RetrievalPolicy
        if device=='cuda':torch.cuda.init()
        return RetrievalPolicy(folder),json.loads((folder/'talos_normalization.json').read_text())
    cfg=ACTConfig.from_pretrained(folder)
    cfg.device=device
    cfg.pretrained_backbone_weights=None
    policy=ACTPolicy.from_pretrained(folder,config=cfg,strict=True)
    policy.to(device).eval()
    return policy,json.loads((folder/'talos_normalization.json').read_text())
