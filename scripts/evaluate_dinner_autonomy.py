"""Headless physical teacher evaluation; not learned-policy submission evidence."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import mujoco
import numpy as np
from scripts.evaluate_dinner_scene import load
from simulation_lab.dinner_autonomy import DinnerSequence
from simulation_lab.scene import HOME,build_scene


def run(seed=42):
    model,data,layout=load(seed)
    for _ in range(200):
        mujoco.mj_step(model,data)
    data.time=0.
    task=DinnerSequence(model,data,layout)
    targets=np.array(HOME*2)
    task.start(kind='set_table')
    ticks=0
    while task.active and ticks < 120000:
        before=data.qpos.copy()
        velocity=data.qvel.copy()
        task.update(targets)
        assert np.array_equal(before,data.qpos), 'Controller changed authoritative positions'
        assert np.array_equal(velocity,data.qvel), 'Controller changed authoritative velocities'
        assert not np.any(data.xfrc_applied) and not np.any(data.qfrc_applied), 'Hidden applied force'
        if not task.active:
            break
        data.ctrl[:]=task.apply_gripper_limit(targets)
        mujoco.mj_step(model,data)
        ticks+=1
    xml,_=build_scene(seed=seed,scenario='dinner')
    return {'seed':seed,'status':task.status,'simulation_seconds':float(data.time),
            'xml_sha256':hashlib.sha256(xml.encode()).hexdigest(),
            'physical_state_writes':0,'external_forces':0,'equality_constraints':model.neq,
            'actuators':model.nu,'task':task.snapshot()}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds',default='42,0,1,2,3,4,5,6,7,8,9')
    parser.add_argument('--output',type=Path,default=Path('.run/dinner-autonomy-evaluation.json'))
    args=parser.parse_args()
    trials=[]
    for seed in map(int,args.seeds.split(',')):
        result=run(seed)
        trials.append(result)
        print(seed,result['status'],result['task']['message'],flush=True)
        report={'schema':'talos.dinner-teacher-evaluation.v1','mujoco':mujoco.__version__,
                'scope':'Exact-state physical teacher. Not a trained policy, Intel benchmark, or official ten-seed video submission.',
                'passed':sum(r['status']=='succeeded' for r in trials),'total':len(trials),'trials':trials}
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    return int(report['passed'] != report['total'])

if __name__=='__main__':
    raise SystemExit(main())
