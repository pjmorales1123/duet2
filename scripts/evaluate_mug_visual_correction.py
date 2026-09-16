"""Freeze and run complete paired physical workflows with an experimental mug."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import mujoco
import numpy as np
import torch
from simulation_lab.engine import LabEngine
from simulation_lab.mug_visual_control import VisualMugSequence
from simulation_lab.storage import require_space


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def space(p,path,expected):
    roots=[ROOT/p[name] for name in ('raw_root','evidence_package')]
    used=sum(f.stat().st_size for root in roots if root.exists() for f in root.rglob('*') if f.is_file())
    if used+expected>p['budget']['maximum_raw_and_packaged_gib']*1024**3:raise ValueError('Physical evidence budget exceeded.')
    return require_space(path,expected,p['budget']['reserve_gib']*1024**3)


def write(p,path,value):
    content=(json.dumps(value,indent=2)+'\n').encode()
    space(p,path,len(content)+1024)
    with Path(path).open('xb') as stream:stream.write(content)


def fingerprint(p,protocol):
    if sha(ROOT/p['baseline_suite'])!=p['baseline_suite_sha256']:raise ValueError('Selected baseline suite changed.')
    interface=ROOT/p['raw_root']/'interface.json'
    result=read(interface)
    if not result['summary']['gate_passed'] or result['protocol_sha256']!=sha(protocol):raise ValueError('Passing frozen interface gate required.')
    for name,digest in result['source_sha256'].items():
        if sha(ROOT/name)!=digest:raise ValueError('A checked interface dependency changed: '+name)
    sources=sorted((ROOT/'simulation_lab').glob('*.py'))
    sources += [Path(__file__),ROOT/'scripts/probe_mug_correction_interface.py',ROOT/'scripts/run_mug_visual_correction.py',
        ROOT/'training_tests/test_mug_visual_control.py']
    files={ROOT/p['baseline_suite']}
    suite=read(ROOT/p['baseline_suite'])
    for relative in suite.values():
        folder=(ROOT/p['baseline_suite']).parent/relative
        files.update(path.resolve() for path in folder.iterdir() if path.is_file())
    for field in ('observer','motor'):
        folder=ROOT/p[field]
        if sha(folder/'manifest.json')!=p[field+'_manifest_sha256']:raise ValueError('Declared model manifest changed.')
        files.add(folder/'manifest.json')
        for name,digest in read(folder/'manifest.json')['files'].items():
            if sha(folder/name)!=digest:raise ValueError('Declared model artifact changed: '+name)
            files.add(folder/name)
    observer_protocol=read(ROOT/p['observer']/'protocol.json')
    camera_protocol=ROOT/observer_protocol['camera_protocol']
    if sha(camera_protocol)!=observer_protocol['camera_protocol_sha256']:raise ValueError('Observer camera protocol changed.')
    files.add(camera_protocol)
    assets=sorted(path for path in (ROOT/'simulation_lab/assets').rglob('*') if path.is_file() and path.suffix not in ('.pyc',))
    if any(not path.resolve().is_relative_to(ROOT) for path in [*sources,*files,*assets]):raise ValueError('A frozen input leaves the repository.')
    return {'protocol_sha256':sha(protocol),'interface_sha256':sha(interface),
        'source_sha256':{path.relative_to(ROOT).as_posix():sha(path) for path in sources},
        'model_sha256':{path.relative_to(ROOT).as_posix():sha(path) for path in sorted(files)},
        'asset_sha256':{path.relative_to(ROOT).as_posix():sha(path) for path in assets}}


def freeze(p,args):
    root=ROOT/p['raw_root'];output=root/'development-freeze.json'
    if output.exists():raise FileExistsError('Preserve the original implementation freeze.')
    if (root/'development').exists() or (root/'evaluation').exists():raise ValueError('Freeze before physical scenes are exposed.')
    inputs=fingerprint(p,args.protocol)
    contract=read(root/'contracts.json')
    if not contract['passed'] or any(sha(ROOT/name)!=digest for name,digest in contract['source_sha256'].items()):
        raise ValueError('Current no-physics controller contracts must pass before freezing.')
    preflight=space(p,output,8*1024**2)
    source_root=root/'frozen-source'
    if source_root.exists():raise FileExistsError('Preserve earlier source freezes.')
    for name,digest in inputs['source_sha256'].items():
        destination=source_root/name
        space(p,destination,(ROOT/name).stat().st_size+1024)
        destination.parent.mkdir(parents=True,exist_ok=True)
        with destination.open('xb') as stream:stream.write((ROOT/name).read_bytes())
        if sha(destination)!=digest:raise ValueError('Source snapshot mismatch.')
    write(p,output,{'schema':p['schema'],'inputs':inputs,'contract_sha256':sha(root/'contracts.json'),
        'development_seeds':p['development_seeds'],'evaluation_seeds':p['evaluation_seeds'],'presets':p['starting_presets'],'modes':p['modes'],
        'preflight':preflight,'parent_git_revision':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()})
    print({'freeze':output.relative_to(ROOT).as_posix(),'source_files':len(inputs['source_sha256']),'models':len(inputs['model_sha256'])},flush=True)


def trial(p,args):
    if args.seed not in p[args.split+'_seeds'] or args.preset not in p['starting_presets'] or args.variant not in p['modes']:
        raise ValueError('Trial lies outside the preregistered cases.')
    root=ROOT/p['raw_root'];freeze_path=root/'development-freeze.json'
    frozen=read(freeze_path)
    inputs=fingerprint(p,args.protocol)
    if inputs!=frozen['inputs']:raise ValueError('The current implementation differs from its frozen development inputs.')
    if args.split=='evaluation':
        selected=read(root/'evaluation-freeze.json')
        if selected['inputs']!=inputs or not selected['development_gate_passed']:raise ValueError('Frozen successful development required before final cases.')
    name=f'{args.seed}-{args.preset}-{args.variant}'
    output=root/args.split/name
    if output.exists():raise FileExistsError('Preserve each attempted physical workflow.')
    preflight=space(p,output,16*1024**2);output.mkdir(parents=True)
    write(p,output/'inputs.json',{'freeze_sha256':sha(freeze_path),'inputs':inputs})
    torch.set_num_threads(2)
    engine=LabEngine(width=320,height=240);engine.dinner_suite=ROOT/p['baseline_suite']
    trace={key:[] for key in ('qpos','qvel','time','stage','targets','neural_progress','correction_count')}
    start=time.perf_counter()
    report={'seed':args.seed,'split':args.split,'preset':args.preset,'variant':args.variant,'instruction':p['instruction'],
        'preflight':preflight,'physics_state_writes_during_control':0,'hidden_forces':0,'equality_constraints':0,
        'first_correction_simulation_time':None,'source_freeze_sha256':sha(freeze_path)}
    try:
        engine._reset(seed=args.seed,scenario='dinner',dinner_preset='task',drawer_open=False,bottle_start=args.preset)
        scene=ET.fromstring(engine.xml);compiler=scene.find('compiler');meshdir=Path(compiler.get('meshdir')).resolve()
        if not meshdir.is_relative_to(ROOT/'simulation_lab/assets'):raise ValueError('Unexpected scene asset path.')
        compiler.set('meshdir',os.path.relpath(meshdir,output).replace('\\','/'))
        with (output/'scene.xml').open('xb') as stream:stream.write(ET.tostring(scene,encoding='utf-8'))
        initial=np.concatenate((engine.data.qpos,engine.data.qvel,engine.data.ctrl,np.asarray([engine.data.time])))
        report['initial_state_sha256']=hashlib.sha256(initial.tobytes()).hexdigest()
        engine._language_command({'text':p['instruction'],'mode':'learned_dinner'})
        plan=engine.task.plan
        expected=['bottle','plate','mug','drawer','fork','spoon'] if args.preset=='upright' else ['reverse_bottle_right','reverse_bottle_left','plate','mug','drawer','fork','spoon']
        report['visual_plan']=plan;report['expected_steps']=expected
        if engine.task.steps!=expected:raise ValueError('Production RGB/language planner selected a different workflow.')
        if args.variant!='baseline':
            # Use the production plan at t=0; change only its future mug child.
            old=engine.task
            q,v,ctrl,target=engine.data.qpos.copy(),engine.data.qvel.copy(),engine.data.ctrl.copy(),engine.target.copy()
            replacement=VisualMugSequence(engine.model,engine.data,engine.layout,old.checkpoints,old.steps,p,args.variant)
            old.close();replacement.plan=plan;engine.task=replacement
            if not all(np.array_equal(a,b) for a,b in ((q,engine.data.qpos),(v,engine.data.qvel),(ctrl,engine.data.ctrl),(target,engine.target))):
                raise RuntimeError('Experimental child-factory construction changed initial physics/control state.')
        limit=int(p['budget']['maximum_simulation_seconds_per_trial']/engine.model.opt.timestep)
        for tick in range(limit):
            if time.perf_counter()-start>p['budget']['maximum_wall_seconds_per_trial']:
                report['budget_stop']='wall_seconds';engine.task.cancel(engine.target);break
            q,v=engine.data.qpos.copy(),engine.data.qvel.copy()
            engine.task.update(engine.target)
            if not np.array_equal(q,engine.data.qpos) or not np.array_equal(v,engine.data.qvel):
                report['physics_state_writes_during_control']+=1;raise RuntimeError('Controller changed qpos/qvel.')
            if engine.model.neq or np.any(engine.data.xfrc_applied) or np.any(engine.data.qfrc_applied):
                report['equality_constraints']=int(engine.model.neq)
                report['hidden_forces']=int(bool(np.any(engine.data.xfrc_applied) or np.any(engine.data.qfrc_applied)))
                raise RuntimeError('Unexpected equality constraint or applied force.')
            child=engine.task.child
            corrections=getattr(child.policy,'corrections',[])
            if corrections and report['first_correction_simulation_time'] is None:
                report['first_correction_simulation_time']=float(engine.data.time)
            if tick%10==0 or not engine.task.active:
                values=(engine.data.qpos.copy(),engine.data.qvel.copy(),float(engine.data.time),child.skill,
                    engine.target.copy(),child.policy.progress,len(corrections))
                for key,value in zip(trace,values):trace[key].append(value)
            if tick%2000==0:space(p,output,16*1024**2)
            if not engine.task.active:break
            engine.data.ctrl[:]=engine.task.apply_gripper_limit(engine.target)
            mujoco.mj_step(engine.model,engine.data)
        if engine.task.active:
            report['budget_stop']='simulation_seconds';engine.task.cancel(engine.target)
        report.update(status=engine.task.status,task=engine.task.snapshot())
    except Exception as exc:
        report.update(status='runtime_error',error_type=type(exc).__name__,message=str(exc))
        if hasattr(engine,'task') and hasattr(engine.task,'snapshot'):report['task']=engine.task.snapshot()
    finally:
        report.update(wall_seconds=time.perf_counter()-start,
            simulation_seconds=float(engine.data.time) if hasattr(engine,'data') else 0.,trace_frames=len(trace['time']))
        space(p,output,16*1024**2)
        with (output/'states.npz').open('xb') as stream:np.savez_compressed(stream,**{key:np.asarray(value) for key,value in trace.items()})
        write(p,output/'report.json',report)
        if hasattr(engine,'task') and hasattr(engine.task,'close'):engine.task.close()
        engine.close()
    mug=next((r for r in report.get('task',{}).get('results',[]) if r.get('skill')=='mug'),None)
    print({'case':name,'status':report['status'],'mug_status':mug.get('status') if mug else None,
        'wall_seconds':report['wall_seconds'],'message':report.get('message',report.get('task',{}).get('message'))},flush=True)
    return 0


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',type=Path,default=ROOT/'docs/robotics/experiments/mug-visual-correction-v1.json')
    parser.add_argument('--mode',choices=['freeze','trial'],required=True)
    parser.add_argument('--split',choices=['development','evaluation'],default='development')
    parser.add_argument('--seed',type=int)
    parser.add_argument('--preset',choices=['upright','wide_left'])
    parser.add_argument('--variant',choices=['baseline','live','frozen'])
    args=parser.parse_args();args.protocol=args.protocol.resolve();p=read(args.protocol)
    if args.mode=='freeze':freeze(p,args)
    else:raise SystemExit(trial(p,args))
