"""Reproducible physical bottle pilot. Reset writes are confined to setup/replay reset.

Examples: python scripts/prepare_bottle_data.py collect
          python scripts/prepare_bottle_data.py render
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys
import threading
import xml.etree.ElementTree as ET
import os

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mujoco
import numpy as np
from simulation_lab.scene import HOME, build_scene, ASSETS
from simulation_lab.dinner_autonomy import DinnerTask
from simulation_lab.recording import export_images
from simulation_lab.storage import require_space, GIB

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'.run/bottle-pilot'
REPORT = ROOT/'docs/robotics/bottle-pilot-results.json'
STATE = mujoco.mjtState.mjSTATE_INTEGRATION

def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=lambda a: a.tolist())+'\n', encoding='utf-8')

def setup(spec):
    xml, layout = build_scene(seed=42, scenario='dinner', dinner_preset='task')
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    data.qpos[:12] = HOME*2
    data.ctrl[:] = HOME*2
    angle = spec['yaw']
    c, s = math.cos(angle/2), math.sin(angle/2)
    quat = np.array([c,-s,c,s])*math.sqrt(.5) if spec['sideways'] else [c,0,0,s]
    adr = model.joint('bottle_free').qposadr[0]
    data.qpos[adr:adr+7] = [spec['x'],spec['y'],layout['table_z']+(.026 if spec['sideways'] else .001),*quat]
    mujoco.mj_forward(model,data)
    bottle=model.body('bottle').id
    initial_penetration=max((max(0.,-float(c.dist)) for c in data.contact
                             if bottle in (model.geom_bodyid[c.geom1],model.geom_bodyid[c.geom2])),default=0.)
    for _ in range(300):
        mujoco.mj_step(model,data)
    data.time = 0.
    # Metadata describes the actual settled reset, without changing the placement guide.
    layout['bottle_reset'] = dict(spec, settled_qpos=data.qpos[adr:adr+7].tolist(),initial_penetration_m=initial_penetration)
    return xml,model,data,layout

def state(model,data):
    result = np.empty(mujoco.mj_stateSize(model,STATE))
    mujoco.mj_getState(model,data,result,STATE)
    return result

def limit(model,data,target,offset,cap):
    result = target.copy()
    i = offset+5
    kp = model.actuator_gainprm[i,0]
    kv = -model.actuator_biasprm[i,2]
    torque = kp*(target[i]-data.qpos[i])-kv*data.qvel[i]
    result[i] = data.qpos[i]+(np.clip(torque,-cap,cap)+kv*data.qvel[i])/kp
    return result

def replay(model, initial, actions, task, hz):
    data = mujoco.MjData(model)
    mujoco.mj_setState(model,data,initial,STATE)  # The only state restoration in replay.
    mujoco.mj_forward(model,data)
    observer = DinnerTask(model,data,task.layout)
    observer._select_item(task.side,task.tube)
    destination = task.destination_position.copy()
    targets = np.array([row['target'] for row in actions])
    if hz == 20:
        indices = np.unique(np.r_[np.arange(0,len(actions),10),len(actions)-1])
        targets = np.column_stack([np.interp(np.arange(len(actions)),indices,targets[indices,i]) for i in range(12)])
    hold = 0.
    unsupported_loss = 0
    contact_gap = longest_gap = 0.
    external_carry_ticks = 0
    for index,row in enumerate(actions):
        forces,lift,up,both = observer._observe()
        carrying=row['stage'] in ('lift','hold','rotate','align','lower') and lift>.012
        contact_gap=contact_gap+.005 if carrying and not both else 0.
        longest_gap=max(longest_gap,contact_gap)
        if row['stage'] in ('rotate','align') and forces['external']>.10:
            external_carry_ticks+=1
        if row['stage'] == 'hold':
            valid = both and lift>.05 and forces['external']<.02
            hold += .005 if valid else 0
            unsupported_loss += int(not valid)
        data.ctrl[:] = limit(model,data,targets[index],task.offset,task.grip_torque)
        mujoco.mj_step(model,data)
    mujoco.mj_forward(model,data)
    error = float(np.linalg.norm(data.body('bottle').xpos[:2]-destination[:2]))
    tilt = float(np.degrees(np.arccos(np.clip(data.body('bottle').xmat[8],-1,1))))
    dof = model.joint('bottle_free').dofadr[0]
    speed = float(np.linalg.norm(data.qvel[dof:dof+3]))
    forces,_,_,_ = observer._observe()
    other_displacement=max(float(np.linalg.norm(data.body(name).xpos-position)) for name,position in observer.others.items())
    ok = (error<.012 and tilt<5 and speed<.003 and forces['base']>.02 and hold>=1.49 and unsupported_loss==0
          and longest_gap<=.18 and external_carry_ticks==0 and other_displacement<.004
          and observer.metrics['other_arm_max_motion_deg']<=1.)
    return {'hz':hz,'passed':bool(ok),'placement_error_m':error,'tilt_deg':tilt,'speed_m_s':speed,
            'table_support_n':forces['base'],'verified_hold_s':hold,'hold_bad_ticks':unsupported_loss,
            'longest_carry_contact_gap_s':longest_gap,'external_carry_ticks':external_carry_ticks,
            'other_object_max_displacement_m':other_displacement,
            'parked_arm_max_motion_deg':observer.metrics['other_arm_max_motion_deg'],
            'state_reset_count':1,'teacher_updates':0}

def trial(spec, keep=False):
    if keep:require_space(OUT, 128*1024**2)
    xml,model,data,layout = setup(spec)
    if layout['bottle_reset']['initial_penetration_m']>.001:
        return {'spec':spec,'outcome':{'status':'failed','stage':'reset','message':'Initial bottle overlap exceeds 1 mm.'},'actions':0}
    initial = state(model,data)
    assert model.neq == 0
    task = DinnerTask(model,data,layout)
    task.start(side=spec.get('arm','auto'),object_id='bottle')
    target = np.array(HOME*2)
    actions,observations = [],[]
    for index in range(25000):
        before,velocity = data.qpos.copy(),data.qvel.copy()
        task.update(target)
        assert np.array_equal(before,data.qpos) and np.array_equal(velocity,data.qvel)
        assert not np.any(data.xfrc_applied) and not np.any(data.qfrc_applied)
        if not task.active:
            break
        ctrl = task.apply_gripper_limit(target)
        if index%10 == 0:
            observations.append(observation(data,index,len(observations)))
        actions.append({'index':index,'time_s':round(index*.005,9),'stage':task.stage,'target':target.copy(),'ctrl':ctrl})
        data.ctrl[:] = ctrl
        mujoco.mj_step(model,data)
    result = {'spec':spec,'outcome':task.snapshot(),'actions':len(actions)}
    if task.active:
        result['outcome']['status']='timeout'
    if task.status == 'succeeded':
        result['replays'] = [replay(model,initial,actions,task,hz) for hz in (200,20)]
    if keep:
        succeeded=task.status=='succeeded'
        folder = OUT/spec['id'] if succeeded else ROOT/'.run/bottle-failures'/(spec['id']+'-'+hashlib.sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest()[:8])
        folder.mkdir(parents=True,exist_ok=False)
        element = ET.fromstring(xml)
        element.find('compiler').set('meshdir',os.path.relpath(ASSETS/'assets',folder).replace('\\','/'))
        ET.ElementTree(element).write(folder/'scene.xml',encoding='unicode')
        np.save(folder/'initial-integration-state.npy',initial)
        observations.append(observation(data,len(actions),len(observations),True))
        for name,rows in [('actions',actions),('observations',observations)]:
            with gzip.open(folder/(name+'.jsonl.gz'),'wt',encoding='utf-8',compresslevel=3) as stream:
                for row in rows:
                    stream.write(json.dumps(row,default=lambda a:a.tolist(),separators=(',',':'))+'\n')
        manifest={'schema_version':2,'id':spec['id'],'engine':'MuJoCo '+mujoco.__version__,
                  'trajectory_complete':True,'layout':layout,'outcome':task.snapshot(),
                  'action_hz':200,'observation_hz':20,'action_count':len(actions),'observation_count':len(observations),
                  'actions':'actions.jsonl.gz','observations':'observations.jsonl.gz','state_model':'scene.xml',
                  'initial_state':{'file':'initial-integration-state.npy','spec':'mjSTATE_INTEGRATION'},
                  'action_contract':'Observations precede commands. Nominal 12 joint position targets; gripper torque saturation at 200 Hz. Replay interpolates nominal 20 Hz targets.',
                  'policy_inputs':['robot_joint_position','robot_joint_velocity','images'],
                  'privileged_fields':['qpos','qvel','actuator_force','layout','outcome','initial_state'],
                  'gripper_torque_cap_nm':getattr(task,'grip_torque',None),'arm':task.side,'replays':result.get('replays',[]),
                  'training_eligible':False,
                  'images':{'status':'pending' if succeeded else 'disabled','hz':20,'resolution':[640,480],'observation_stride':1,
                            'cameras':['overhead','left_wrist_cam','right_wrist_cam']}}
        save(folder/'manifest.json',manifest)
        result['folder']=folder.relative_to(ROOT).as_posix()
    return result

def observation(data,action_index,index,terminal=False):
    return {'index':index,'action_index':action_index,'time_s':round(action_index*.005,9),
            'simulation_time_s':float(data.time),'terminal':terminal,'qpos':data.qpos.copy(),'qvel':data.qvel.copy(),
            'robot_joint_position':data.qpos[:12].copy(),'robot_joint_velocity':data.qvel[:12].copy(),
            'actuator_force':data.actuator_force.copy()}

def collect():
    require_space(OUT, GIB)
    OUT.mkdir(parents=True,exist_ok=True)
    specs=[]
    for sideways in (False,True):
        # Small pose perturbations around physically demonstrated approach families.
        anchors = [(-.10,-.15,1),(-.05,-.175,3),(-.05,-.15,5),(0.,-.15,5)] if sideways else [(.025,-.075,0)]
        for delta in (0.,.005,-.005,.01,-.01,.015,-.015):
            for x,y,a in anchors:
                specs.append({'x':x+delta,'y':y,'yaw':a*math.pi/4+delta*2,'sideways':sideways})
    validation=[{'x':-.075,'y':-.14,'yaw':math.pi/4,'sideways':True},
                {'x':.025,'y':-.065,'yaw':.3,'sideways':False},
                {'x':-.04,'y':-.16,'yaw':5*math.pi/4,'sideways':True}]
    save(OUT/'reserved-validation.json',{'usage':'Reserved; do not use for demonstrations or tuning. Not yet evaluated.','poses':validation})
    results=[]
    counts={False:0,True:0}
    # Additional upright y/headings to reach ten distinct starts.
    specs += [{'x':.025+dx,'y':-.075+dy,'yaw':yaw,'sideways':False}
              for dx,dy,yaw in [(0,.005,.2),(0,-.005,.4),(.005,.005,.6),(-.005,-.005,.8),(0,0,1.)]]
    for spec in specs:
        mode=spec['sideways']
        if counts[mode]>=10:
            continue
        spec['id']=('sideways' if mode else 'upright')+f'-{counts[mode]+1:02d}'
        result=trial(spec,keep=True)
        results.append(result)
        if result['outcome']['status']=='succeeded':
            counts[mode]+=1
        print(spec,result['outcome']['status'],[r['passed'] for r in result.get('replays',[])],flush=True)
        save(REPORT,{'counts':{'upright':counts[False],'sideways':counts[True]},'attempts':results,'reserved_validation':validation})
    assert counts=={False:10,True:10},counts

def render_one(folder):
    export_images(str(folder),threading.Event())
    return folder.name,json.loads((folder/'manifest.json').read_text())['images']

def render(pattern='*'):
    folders=[]
    for file in sorted(OUT.glob(pattern+'/manifest.json')):
        manifest=json.loads(file.read_text())
        if manifest['images']['status']=='completed':
            continue
        folders.append(file.parent)
    with ProcessPoolExecutor(max_workers=4) as pool:
        for result in pool.map(render_one,folders):
            print(result,flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['collect','render'])
    parser.add_argument('--pattern',default='*')
    args=parser.parse_args()
    collect() if args.action=='collect' else render(args.pattern)
