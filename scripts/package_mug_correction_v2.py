"""Package both motor candidates and reproduce the callable CPU interface."""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import mujoco
import numpy as np
from scripts.mug_correction_motor_v1 import read,sha,space,write,data_gate
from scripts.mug_correction_motor_v2 import frozen,aggregate
from simulation_lab.mug_correction_runtime import LocalMugMotor


def verify(p):
    model_root,training,evidence=[ROOT/p[key] for key in ('model_package','training_package','evidence_package')]
    checked=0
    for root in (model_root,training,evidence):
        if (root/'manifest.json').exists():
            for name,digest in read(root/'manifest.json')['files'].items():
                if sha(root/name)!=digest:raise ValueError('Packaged artifact changed: '+name)
                checked+=1
    model=mujoco.MjModel.from_xml_path(str(ROOT/p['scene']))
    runtime=LocalMugMotor(model_root,model)
    with np.load(training/'evaluation/data.npz',allow_pickle=False) as archive:
        q,d=archive['joints'].copy(),archive['requested_translation'].copy()
    expected=read(evidence/'fresh-kinematics.json')
    rows,differences=[],[]
    for index,(joint,requested) in enumerate(zip(q,d)):
        row=runtime.predict(joint,requested)
        original=expected['rows'][index]
        differences.append(float(np.max(abs(np.asarray(row['joint_delta_rad'])-original['joint_delta_rad']))))
        if row['accepted']!=original['accepted']:raise ValueError('Callable runtime acceptance differs from evaluated export.')
        row['index']=index;rows.append(row)
    summary=aggregate(rows,p)
    passed=summary['gate_passed'] and max(differences)<=p['export']['maximum_joint_delta_parity_rad']
    if not passed:raise ValueError('Callable packaged runtime failed reproduction.')
    return {'schema':p['schema'],'passed':True,'checked_manifest_files':checked,'states':len(rows),
        'maximum_callable_joint_delta_difference_rad':max(differences),'all_difference_rad':differences,'summary':summary,
        'device':runtime.device,'raw_run_files_read':0,'new_physical_trials':0,
        'scope':'Packaged-data and callable CPU reproduction of the already exposed fresh kinematic set; no new generalization or contact physics.'}


def package(args):
    p=read(args.protocol)
    if args.verify_only:
        result=verify(p)
        print({k:result[k] for k in ('passed','checked_manifest_files','states','maximum_callable_joint_delta_difference_rad','summary')},flush=True)
        return
    raw=ROOT/p['raw_root']
    roots={key:ROOT/p[key+'_package'] for key in ('model','training','evidence')}
    if any(root.exists() for root in roots.values()):raise FileExistsError('Preserve previous packages.')
    fit,checkpoint=frozen(p,args)
    parity,fresh=read(raw/'openvino/parity.json'),read(raw/'evaluation/evaluation.json')
    if not parity['passed'] or not fresh['passed'] or fresh['checkpoint_sha256']!=sha(checkpoint):
        raise ValueError('Successful unchanged export and fresh gates required.')
    for name,digest in fresh['ir_sha256'].items():
        if sha(raw/'openvino'/name)!=digest:raise ValueError('Evaluated IR changed.')
    if [c['step'] for c in fit['candidates']]!=p['training']['checkpoints']:raise ValueError('Both checkpoints required.')
    payload={key:[] for key in roots};source_hashes=fit['source_sha256'].copy()
    for split in ('training','development','evaluation'):
        manifest,arrays=data_gate(p,raw/split,args.protocol)
        audit=read(raw/split/'audit.json')
        if not audit['passed'] or audit['data_sha256']!=sha(raw/split/'data.npz'):raise ValueError('Complete independent geometry audits required.')
        source_hashes.update(manifest['source_sha256'])
        for name in ('data.npz','manifest.json','audit.json'):
            payload['training'].append((raw/split/name,Path(split)/name))
    for c in fit['candidates']:
        folder=raw/f'fit/step-{c["step"]:06d}'
        if sha(folder/'model.safetensors')!=c['checkpoint_sha256']:raise ValueError('Candidate changed.')
        original=read(folder/'development.json')
        if original['summary']!=c['development']:raise ValueError('Checkpoint development scores changed.')
        payload['model'].append((folder/'model.safetensors',Path(folder.name)/'model.safetensors'))
        payload['evidence'].append((folder/'development.json',Path(folder.name)/'development.json'))
    for name in ('motor.xml','motor.bin'):
        payload['model'].append((raw/'openvino'/name,Path('openvino')/name))
    for source,name in (('fit/training.json','training.json'),('openvino/parity.json','parity.json'),('evaluation/evaluation.json','fresh-kinematics.json')):
        payload['evidence'].append((raw/source,Path(name)))
    for path in sorted((raw/'attempts').rglob('*')):
        if path.is_file():payload['evidence'].append((path,path.relative_to(raw)))
    for path in sorted((ROOT/'.run/final-goal').glob('mug-correction-motor-v2-*.log')):
        if path.name not in ('mug-correction-motor-v2-package.log','mug-correction-motor-v2-package-verify.log'):
            payload['evidence'].append((path,Path('console')/path.name))
    for helper in (Path(__file__),ROOT/'simulation_lab/mug_correction_runtime.py',ROOT/'scripts/collect_mug_correction_fresh.py'):
        source_hashes[helper.relative_to(ROOT).as_posix()]=sha(helper)
    for name,digest in source_hashes.items():
        if sha(ROOT/name)!=digest:raise ValueError('Frozen dependency changed: '+name)
        if name.endswith('.py'):payload['training'].append((ROOT/name,Path('source')/name))
    for key in roots:payload[key].append((args.protocol,Path('protocol.json')))
    preflight=space(p,roots['evidence'],sum(source.stat().st_size for items in payload.values() for source,_ in items)+8*1024**2)
    for root in roots.values():root.mkdir(parents=True)
    console={}
    for key,items in payload.items():
        for source,name in items:
            target=roots[key]/name;space(p,target,source.stat().st_size+1024);target.parent.mkdir(parents=True,exist_ok=True)
            content=source.read_bytes()
            if name.parts[0]=='console':
                for prefix,label in ((str(ROOT),'<repository>'),(ROOT.as_posix(),'<repository>'),(sys.prefix,'<environment>'),(sys.base_prefix,'<python>')):
                    content=content.replace(prefix.encode(),label.encode())
                console[name.as_posix()]={'original_sha256':sha(source),'original_bytes':source.stat().st_size,'private_paths_redacted':content!=source.read_bytes()}
            with target.open('xb') as stream:stream.write(content)
    write(p,roots['model']/'runtime.json',{'schema':'talos.mug-joint-limited-runtime.v1','protocol_sha256':sha(args.protocol),
        'selected_step':fit['selected']['step'],'checkpoint_sha256':sha(checkpoint),'weights':f'step-{fit["selected"]["step"]:06d}/model.safetensors',
        'ir_sha256':fresh['ir_sha256'],'support':fit['support'],'guard':p['runtime_guard'],'step_sizing':p['step_sizing'],
        'scope':'Measured joints and requested local XY translation only; neural delta, explicit joint limiting, forward-only refusal guard.'})
    write(p,roots['training']/'reproduction-inputs.json',{'schema':p['schema'],'source_sha256':source_hashes,
        'all_states':{'training':8192,'development':1024,'evaluation':1024},'prior_rejections_retained':True,
        'original_requested_and_effective_translations_retained':True,'source_note':'The fitted V2 source stays frozen, including its fresh-dispatch bug. collect_mug_correction_fresh.py is the corrected standalone fresh entrypoint; both launch incidents and console provenance are preserved.'})
    reproduction=verify(p)
    write(p,roots['evidence']/'callable-reproduction.json',reproduction)
    write(p,roots['evidence']/'package-audit.json',{'schema':p['schema'],'protocol_sha256':sha(args.protocol),'source_sha256':sha(Path(__file__)),
        'preflight':preflight,'console_provenance':console,'fresh':fresh['summary'],'callable_reproduction_passed':reproduction['passed'],
        'both_checkpoints_retained':True,'selected_dinner_models_unchanged':True,'new_physical_trials':0,'motor_integration':False})
    counts={}
    for key,root in roots.items():
        files={path.relative_to(root).as_posix():sha(path) for path in sorted(root.rglob('*')) if path.is_file()}
        size=sum((root/name).stat().st_size for name in files)
        write(p,root/'manifest.json',{'schema':p['schema'],'files':files,'bytes':size})
        counts[key]={'files':len(files),'bytes':size}
    print({'packages':counts,'fresh':fresh['summary'],'callable_reproduction':{k:reproduction[k] for k in ('passed','states','maximum_callable_joint_delta_difference_rad')}},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',type=Path,default=ROOT/'docs/robotics/experiments/mug-correction-motor-v2.json')
    parser.add_argument('--verify-only',action='store_true')
    args=parser.parse_args();args.protocol=args.protocol.resolve();package(args)
