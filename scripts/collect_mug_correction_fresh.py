"""Collect the once-only V2 fresh split after its frozen launcher dispatch error.

The original fit/collector source is retained byte-for-byte. This adapter calls
its unchanged sampling function and applies the same declared local step sizing.
"""
import argparse
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import mujoco
import numpy as np
from scripts.mug_correction_motor_v1 import read,sha,space,write
from scripts.mug_correction_motor_v2 import frozen,sources,fresh_inputs
from simulation_lab.mug_correction_motor import MugRobotGeometry,assess_delta
from simulation_lab.mug_correction_limits import limited_delta,assess_limited


def collect(args):
    p=read(args.protocol)
    if p['schema']!='talos.mug-correction-motor.v2':raise ValueError('This adapter is specific to the frozen V2 protocol.')
    raw=ROOT/p['raw_root'];folder=raw/'evaluation'
    if folder.exists():raise FileExistsError('Preserve every fresh attempt.')
    fit,checkpoint=frozen(p,args)
    parity=read(raw/'openvino/parity.json')
    if not parity['passed'] or parity['protocol_sha256']!=sha(args.protocol) or parity['checkpoint_sha256']!=sha(checkpoint):
        raise ValueError('Unchanged passing selection/export required.')
    for name,digest in parity['ir_sha256'].items():
        if sha(raw/'openvino'/name)!=digest:raise ValueError('Frozen IR changed.')
    source_hashes=sources(p,args.protocol)
    source_hashes[Path(__file__).relative_to(ROOT).as_posix()]=sha(Path(__file__))
    preflight=space(p,folder,32*1024**2)
    started=time.perf_counter()
    model=mujoco.MjModel.from_xml_path(str(ROOT/p['scene']));geometry=MugRobotGeometry(model)
    data=fresh_inputs(p,model)
    count=p['data']['evaluation_states'];original=data['translation'].copy()
    data['requested_translation']=original;data['translation']=np.zeros((count,3),float)
    data['labels']=np.zeros((count,5),float);data['accepted']=np.zeros(count,bool);data['step_fraction']=np.zeros(count,float)
    lo,hi=model.actuator_ctrlrange[6:11].T;support={'minimum':lo,'maximum':hi}
    guard={**p['runtime_guard'],'maximum_position_error_mm':p['data']['maximum_label_position_error_mm']}
    rows=[]
    for i,(q,matrix,requested) in enumerate(zip(data['joints'],data['coefficients'],original)):
        raw_delta=matrix@requested
        delta,effective,fraction=limited_delta(raw_delta,requested,p['step_sizing']['maximum_joint_delta_rad'])
        data['labels'][i]=q.astype(float)+delta;data['translation'][i]=effective;data['step_fraction'][i]=fraction
        row=assess_delta(geometry,q,effective,data['labels'][i]-q,support,guard)
        command=assess_limited(geometry,q,requested,raw_delta,support,guard,p['step_sizing'])
        row.update(index=i,step_fraction=fraction,original_requested_translation_m=requested.tolist(),accepted=command['accepted'],
                   episode_index=int(data['episode_index'][i]),nominal_seconds=float(data['nominal_seconds'][i]))
        data['accepted'][i]=row['accepted'];rows.append(row)
        if i%512==0:
            space(p,folder,16*1024**2)
            if time.perf_counter()-started>p['budget']['maximum_generation_minutes_per_split']*60:raise TimeoutError('Declared generation time exceeded.')
    space(p,folder,32*1024**2);folder.mkdir()
    with (folder/'data.npz').open('xb') as stream:np.savez_compressed(stream,**data)
    result={'schema':p['schema'],'protocol_sha256':sha(args.protocol),'source_sha256':source_hashes,'origin':None,'split':'evaluation',
        'states':count,'accepted_labels':int(data['accepted'].sum()),'label_coverage_passed':float(data['accepted'].mean())>=p['data']['minimum_label_coverage'],
        'step_fraction':{'minimum':float(data['step_fraction'].min()),'median':float(np.median(data['step_fraction']))},
        'data_sha256':sha(folder/'data.npz'),'rows':rows,'preflight':preflight,'wall_seconds':time.perf_counter()-started,
        'parent_git_revision':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'collector_adapter':Path(__file__).relative_to(ROOT).as_posix(),
        'scope':'Fresh offline kinematics from unchanged V2 sampling/step-sizing functions. No contact physics or controller integration.'}
    write(p,folder/'manifest.json',result)
    print({k:result[k] for k in ('states','accepted_labels','label_coverage_passed','step_fraction','wall_seconds')},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',type=Path,default=ROOT/'docs/robotics/experiments/mug-correction-motor-v2.json')
    args=parser.parse_args();args.protocol=args.protocol.resolve();collect(args)
