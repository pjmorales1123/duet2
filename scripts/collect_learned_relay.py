"""Physical relay demonstrations: separate learned legs with explicit destinations.

First release on the shared table and park; then let the other arm regrasp.
The collector never describes this as a direct hand-to-hand exchange.
"""
import argparse
from copy import deepcopy
import json
import os
import re
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import mujoco
import numpy as np
from simulation_lab.scene import build_scene,HOME
from simulation_lab.storage import require_space
from scripts.collect_dinner_learning import collect_episode,save_json


def validate_protocol(protocol):
    specs,legs=protocol['training'],protocol['legs']
    if not 1<=len(specs)<=32 or len({s['seed'] for s in specs})!=len(specs):raise ValueError('Use 1–32 unique training seeds.')
    if len(legs)!=2 or len({l['name'] for l in legs})!=2:raise ValueError('A relay needs two distinct learned legs.')
    for leg in legs:
        if not re.fullmatch(r'(relay|reverse)_bottle_(left|right)',leg['name']) or leg['arm'] not in ('left','right'):
            raise ValueError('Invalid relay leg name or arm.')
        point=leg.get('destination_xy')
        if point is not None and (np.shape(point)!=(2,) or not np.isfinite(point).all() or np.any(np.abs(point)>.4)):
            raise ValueError('Invalid fixed relay destination.')
    for spec in specs:
        if type(spec['seed']) is not int or not 0<=spec['seed']<=2147483647:raise ValueError('Invalid seed.')
        if spec.get('pose'):
            point=spec['pose']
            if set(point)!={'x','y','yaw'} or not np.isfinite(list(point.values())).all():raise ValueError('Invalid explicit bottle pose.')
            if abs(point['x'])>.3 or abs(point['y'])>.3 or abs(point['yaw'])>np.pi:raise ValueError('Bottle pose outside configured bounds.')


def collect(spec,output,legs):
    require_space(output,256*1024**2)
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True)
    xml,layout=build_scene(seed=spec['seed'],scenario='dinner',dinner_preset='task')
    model=mujoco.MjModel.from_xml_string(xml);model.vis.quality.offsamples=0
    data=mujoco.MjData(model);data.qpos[:12]=HOME*2;data.ctrl[:]=HOME*2
    if spec.get('pose'):
        p=spec['pose'];address=model.joint('bottle_free').qposadr[0]
        data.qpos[address:address+7]=[p['x'],p['y'],layout['table_z']+.001,np.cos(p['yaw']/2),0,0,np.sin(p['yaw']/2)]
    mujoco.mj_forward(model,data)
    for _ in range(300):mujoco.mj_step(model,data)
    data.time=0.
    scene=ET.fromstring(xml);compiler=scene.find('compiler')
    compiler.set('meshdir',os.path.relpath(compiler.get('meshdir'),output).replace('\\','/'))
    (output/'scene.xml').write_text(ET.tostring(scene,encoding='unicode'),encoding='utf-8')
    rows=[]
    for leg in legs:
        current=deepcopy(layout)
        if leg.get('destination_xy') is not None:
            current['targets']=[t for t in current['targets'] if t['object_id']!='bottle']
            current['targets'].append({'id':leg['name']+'_destination','object_id':'bottle',
                'position_m':[*leg['destination_xy'],layout['table_z']],'radius_m':.012})
        result=collect_episode(model,data,current,'bottle',output/leg['name'],side=leg['arm'])
        rows.append({'leg':leg,**result})
        save_json(output/'summary.json',{'spec':spec,'legs':rows,'eligible':all(r['training_eligible'] for r in rows) and len(rows)==len(legs)})
        print(json.dumps({'seed':spec['seed'],'leg':leg['name'],'arm':result['arm'],
                          'teacher':result['teacher_status'],'replay':result['training_eligible'],'message':result['message']}),flush=True)
        if not result['training_eligible']:break
    return len(rows)==len(legs) and all(r['training_eligible'] for r in rows)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();protocol=json.loads(a.protocol.read_text(encoding='utf-8'));validate_protocol(protocol)
    if a.output.exists():raise FileExistsError(a.output)
    specs=protocol['training'];require_space(a.output,len(specs)*256*1024**2)
    if len({s['seed'] for s in specs})!=len(specs):raise ValueError('Duplicate training seed.')
    a.output.mkdir(parents=True);save_json(a.output/'protocol.json',protocol)
    passed=[collect(s,a.output/f"seed-{s['seed']}",protocol['legs']) for s in specs]
    save_json(a.output/'summary.json',{'passed':sum(passed),'attempted':len(passed),'purpose':'Teacher and exact replay data, not learned execution'})
    raise SystemExit(int(not all(passed)))
