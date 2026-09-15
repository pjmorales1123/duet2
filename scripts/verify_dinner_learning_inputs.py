"""Replay the actual stage-aligned training targets through independent physics."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import mujoco
import numpy as np
from simulation_lab.dinner_monitor import DinnerPhysicalMonitor
from simulation_lab.policy_control import apply_targets
from simulation_lab.storage import require_space
from scripts.prepare_bottle_data import STATE


def verify(source,datasets,output,seeds=None):
    if output.exists():raise FileExistsError(output)
    require_space(output,16*1024**2)
    meta=json.loads((source/'retrieval.json').read_text())
    digest=hashlib.sha256((source/'retrieval.npz').read_bytes()).hexdigest()
    skill=meta.get('logical_skill',meta['skill'])
    with np.load(source/'retrieval.npz',allow_pickle=False) as z:
        arrays={k:z[k] for k in ('bounds','actions','seconds')}
    rows=[]
    for i,name in enumerate(meta['episodes']):
        if seeds and int(name.removeprefix('seed-')) not in seeds:continue
        require_space(output,16*1024**2)
        folders=[root/name/skill for root in datasets if (root/name/skill/'manifest.json').is_file()]
        if len(folders)!=1:raise ValueError('Require exactly one source episode: '+name)
        folder=folders[0];manifest=json.loads((folder/'manifest.json').read_text())
        lineage=next(row for row in meta['lineage'] if row['episode']==name)
        if hashlib.sha256((folder/'manifest.json').read_bytes()).hexdigest()!=lineage['manifest_sha256']:
            raise ValueError('Episode manifest changed: '+name)
        model=mujoco.MjModel.from_xml_path(str(folder.parent/'scene.xml'));data=mujoco.MjData(model)
        mujoco.mj_setState(model,data,np.load(folder/'initial-integration-state.npy',allow_pickle=False),STATE)
        mujoco.mj_forward(model,data)
        layout=deepcopy(manifest['layout'])
        if 'trained_destination_m' in meta:
            layout['targets']=[t for t in layout['targets'] if t['object_id']!=meta['skill']]
            layout['targets'].append({'id':skill+'_goal','object_id':meta['skill'],'position_m':meta['trained_destination_m']})
        side='left' if meta['arm_offset']==0 else 'right'
        monitor=DinnerPhysicalMonitor(model,data,layout,meta['skill'],side)
        a,b=arrays['bounds'][i];actions=arrays['actions'][a:b];seconds=arrays['seconds'][a:b]
        began=time.perf_counter()
        for tick in range(int(np.ceil((seconds[-1]+4)/model.opt.timestep))):
            if monitor.update() or monitor.succeeded:break
            t=tick*model.opt.timestep
            target=np.array([np.interp(t,seconds,actions[:,j]) for j in range(12)])
            data.ctrl[:]=apply_targets(model,data,target,meta['gripper_cap_nm'],meta['arm_offset'])
            mujoco.mj_step(model,data)
            assert model.neq==0 and not np.any(data.xfrc_applied) and not np.any(data.qfrc_applied)
        monitor.update()
        row={'episode':name,**monitor.report(),'wall_seconds':time.perf_counter()-began}
        rows.append(row)
        print(json.dumps({'episode':name,'skill':skill,'passed':row['passed'],'failure':row['failure'],
                          'placement_error_mm':row['metrics'].get('placement_error_mm')}),flush=True)
    complete=len(rows)==len(meta['episodes'])
    report={'schema':'talos.aligned-input-replay.v1','source_sha256':digest,'skill':skill,
            'complete':complete,'passed':complete and all(row['passed'] for row in rows),
            'episodes':rows,'teacher_updates':0,'scope':'Physical replay of aligned training labels, not neural inference.'}
    require_space(output,16*1024**2)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    return all(row['passed'] for row in rows) and bool(rows)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--dataset',type=Path,action='append',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seeds',help='Optional diagnostic subset; cannot satisfy the training gate.')
    a=p.parse_args()
    raise SystemExit(0 if verify(a.source,a.dataset,a.output,{int(s) for s in a.seeds.split(',')} if a.seeds else None) else 1)
