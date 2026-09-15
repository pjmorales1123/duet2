"""Strict policy-facing view of verified pilot data; privileged state stays out."""
import gzip
import json
from pathlib import Path
import numpy as np


def load_policy_episode(folder):
    folder = Path(folder).resolve()
    exclusions=folder.parent/'excluded.json'
    if exclusions.exists() and folder.name in json.loads(exclusions.read_text(encoding='utf-8')):
        raise ValueError('This episode was excluded during reset-quality review.')
    manifest = json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
    if not manifest.get('training_eligible') or manifest['images']['status'] != 'completed':
        raise ValueError('A complete, verified episode with images is required.')
    if manifest['images']['hz'] != 20:
        raise ValueError('This pilot loader requires synchronized 20 Hz images.')
    with gzip.open(folder/manifest['actions'],'rt',encoding='utf-8') as f:
        actions = [json.loads(line) for line in f]
    with gzip.open(folder/manifest['observations'],'rt',encoding='utf-8') as f:
        observations = [json.loads(line) for line in f if line.strip()]
    image_index={}
    for line in (folder/manifest['images']['index']).read_text().splitlines():
        row=json.loads(line)
        path=(folder/row['path']).resolve()
        if not path.is_relative_to(folder) or not path.is_file():
            raise ValueError('Invalid image reference.')
        image_index[row['observation_index'],row['camera']]=str(path)
    samples=[]
    for obs in observations:
        if obs['terminal']:
            continue
        index=obs['action_index']
        sample={'time_s':obs['time_s'],
                'joint_position':np.asarray(obs['robot_joint_position'],dtype=np.float32),
                'joint_velocity':np.asarray(obs['robot_joint_velocity'],dtype=np.float32),
                'action':np.asarray(actions[index]['target'],dtype=np.float32),
                'images':{camera:image_index[obs['index'],camera] for camera in manifest['images']['cameras']}}
        if any(sample[key].shape!=(12,) or not np.isfinite(sample[key]).all()
               for key in ('joint_position','joint_velocity','action')):
            raise ValueError('Invalid robot sample.')
        if abs(actions[index]['time_s']-obs['time_s'])>1e-8:
            raise ValueError('Misaligned observation and target.')
        samples.append(sample)
    return samples
