"""Recount every paired mug workflow and check frozen inputs and trace evidence."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
import mujoco
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.evaluate_mug_visual_correction import read,sha,space,write,fingerprint
from scripts.package_spoon_release import physical_criteria
from simulation_lab.mug_visual_geometry import correction_request
from simulation_lab.mug_correction_runtime import LocalMugMotor


def audit(p,protocol,root,split):
    frozen=read(root/'development-freeze.json')
    if frozen['inputs']['protocol_sha256']!=sha(protocol):raise ValueError('The physical protocol changed.')
    for name,digest in frozen['inputs']['source_sha256'].items():
        if sha(root/'frozen-source'/name)!=digest:raise ValueError('A frozen source snapshot changed: '+name)
    for kind in ('model_sha256','asset_sha256'):
        for name,digest in frozen['inputs'][kind].items():
            if sha(ROOT/name)!=digest:raise ValueError('An evaluated model or asset changed: '+name)
    batch=read(root/(split+'-batch.json'))
    expected={(s,preset,mode) for s in p[split+'_seeds'] for preset in p['starting_presets'] for mode in p['modes']}
    actual=[(r['seed'],r['preset'],r['variant']) for r in batch['rows']]
    if len(actual)!=len(expected) or set(actual)!=expected or batch['completed']!=len(expected):
        raise ValueError('Every declared workflow must be retained before aggregation.')
    counts={mode:{preset:0 for preset in p['starting_presets']} for mode in p['modes']}
    mugs=json.loads(json.dumps(counts));rows=[];records={};traces={};frames=0;query_total=0;issued=0
    motor=None;max_motor_difference=0.
    for entry in batch['rows']:
        key=(entry['seed'],entry['preset'],entry['variant'])
        name=f'{key[0]}-{key[1]}-{key[2]}';folder=root/split/name
        r=read(folder/'report.json');task=r.get('task',{});results=task.get('results',[])
        if (r['seed'],r['preset'],r['variant'])!=key or r['split']!=split:raise ValueError('A case identity changed.')
        if r['source_freeze_sha256']!=sha(root/'development-freeze.json'):raise ValueError('A case references another implementation.')
        if read(folder/'inputs.json')['inputs']!=frozen['inputs']:raise ValueError('A case used different inputs.')
        if r['physics_state_writes_during_control'] or r['hidden_forces'] or r['equality_constraints']:
            raise ValueError('A controller-assistance invariant failed.')
        passed=r['status']=='succeeded' and task.get('completed_steps')==r.get('expected_steps')
        if passed!=(entry['status']=='succeeded') or entry['exit_code']!=0:raise ValueError('The batch disagrees with its report or launcher.')
        for outcome in results:physical_criteria(outcome)
        mug=next((v for v in results if v['skill']=='mug'),None)
        mug_passed=bool(mug and mug['status']=='succeeded')
        counts[key[2]][key[1]]+=int(passed);mugs[key[2]][key[1]]+=int(mug_passed)
        with np.load(folder/'states.npz',allow_pickle=False) as archive:trace={k:archive[k] for k in archive.files}
        n=len(trace['time']);frames+=n
        if not n or n!=r['trace_frames'] or any(len(a)!=n for a in trace.values()) or np.any(np.diff(trace['time'])<=0):
            raise ValueError('A complete synchronized monotonic state trace is required.')
        if any(not np.isfinite(a).all() for k,a in trace.items() if k!='stage'):raise ValueError('Nonfinite state trace.')
        model=mujoco.MjModel.from_xml_path(str((folder/'scene.xml').resolve()))
        if model.neq:raise ValueError('A portable scene contains equality constraints.')
        if trace['qpos'].shape!=(n,model.nq) or trace['qvel'].shape!=(n,model.nv):raise ValueError('Trace dimensions disagree with its scene.')
        corrections=[] if mug is None else mug.get('mug_visual_corrections',[])
        if key[2]=='baseline' and corrections:raise ValueError('Baseline unexpectedly contains corrections.')
        previous_phase=None;offset=np.zeros(5);frozen_rgb=None
        for correction in corrections:
            query_total+=1;phase=correction['nominal_seconds']
            if not p['control']['correction_nominal_seconds'][0]-1e-9<=phase<=p['control']['correction_nominal_seconds'][1]+1e-9:
                raise ValueError('A correction left its declared phase interval.')
            if previous_phase is not None and phase<=previous_phase:raise ValueError('Waiting accumulated another correction.')
            previous_phase=phase
            used=correction.get('used_rgb_sha256');current=correction.get('current_rgb_sha256')
            if used is not None:
                if key[2]=='live' and used!=current:raise ValueError('Live correction did not use current image bytes.')
                if key[2]=='frozen':
                    if frozen_rgb is None:frozen_rgb=current
                    if used!=frozen_rgb:raise ValueError('Frozen correction image bytes changed.')
            observation=correction.get('observation',{})
            if observation.get('status')=='observed' and 'motor' in correction:
                origin,request=correction_request(observation,p['control'])
                if not np.allclose(origin,correction['estimated_origin_m'],rtol=0,atol=1e-14) or not np.allclose(request,correction['requested_translation_m'],rtol=0,atol=1e-14):
                    raise ValueError('The visual request disagrees with its observed semantic points.')
                if motor is None:motor=LocalMugMotor(ROOT/p['motor'],model)
                original=correction['motor'];reproduced=motor.predict(original['joints'],request)
                for field,value in original.items():
                    if field=='neural_inference_ms':continue
                    if isinstance(value,bool):
                        if value!=reproduced[field]:raise ValueError('A motor query refusal changed.')
                    elif isinstance(value,(float,int,list)):
                        difference=float(np.max(abs(np.asarray(value)-np.asarray(reproduced[field]))))
                        max_motor_difference=max(max_motor_difference,difference)
                        if difference>1e-9:raise ValueError('A motor query cannot be independently reproduced.')
                    elif value!=reproduced[field]:raise ValueError('A motor query refusal changed.')
                if correction['status']=='issued':
                    if not original['accepted']:raise ValueError('A refused local motor step was issued.')
                    offset+=np.asarray(original['joint_delta_rad']);issued+=1
                    if np.max(abs(offset))>p['control']['maximum_cumulative_joint_offset_rad']+1e-12:
                        raise ValueError('A correction exceeded the cumulative offset limit.')
                    if not np.allclose(offset,correction['joint_offset_rad'],rtol=0,atol=1e-12):raise ValueError('Accumulated offset disagrees.')
        row={'seed':key[0],'preset':key[1],'variant':key[2],'status':r['status'],'passed':passed,'mug_passed':mug_passed,
            'mug_status':None if mug is None else mug['status'],'mug_error_mm':None if mug is None else mug.get('metrics',{}).get('placement_error_mm'),
            'message':r.get('message',task.get('message')),'simulation_seconds':r['simulation_seconds'],'wall_seconds':r['wall_seconds'],
            'trace_frames':n,'correction_queries':len(corrections),'issued_corrections':sum(c['status']=='issued' for c in corrections),
            'first_correction_simulation_time':r['first_correction_simulation_time'],
            'report':(Path(split)/name/'report.json').as_posix(),'report_sha256':sha(folder/'report.json'),
            'states_sha256':sha(folder/'states.npz'),'scene_sha256':sha(folder/'scene.xml')}
        records[key]=r;traces[key]=trace;rows.append(row)
    pairs=[];reductions=[]
    row_lookup={(r['seed'],r['preset'],r['variant']):r for r in rows}
    for seed in p[split+'_seeds']:
        for preset in p['starting_presets']:
            keys=[(seed,preset,mode) for mode in p['modes']]
            initial=[records[k].get('initial_state_sha256') for k in keys]
            times=[records[k]['first_correction_simulation_time'] for k in keys if records[k]['first_correction_simulation_time'] is not None]
            stop=min(times) if times else float('inf')
            views=[traces[k] for k in keys];masks=[v['time']<stop-1e-9 for v in views]
            mismatches=[field for field in ('time','qpos','qvel','stage','targets','neural_progress')
                if any(not np.array_equal(views[0][field][masks[0]],v[field][mask]) for v,mask in zip(views[1:],masks[1:]))]
            pairs.append({'seed':seed,'preset':preset,'initial_match':bool(all(initial) and len(set(initial))==1),
                'prefix_match':not mismatches,'compared_frames':[int(m.sum()) for m in masks],
                'before_simulation_seconds':None if not times else stop,'different_fields':mismatches})
            live,frozen_row=[row_lookup[(seed,preset,mode)] for mode in ('live','frozen')]
            if live['mug_passed'] and frozen_row['mug_passed']:
                reductions.append({'seed':seed,'preset':preset,'frozen_minus_live_error_mm':frozen_row['mug_error_mm']-live['mug_error_mm']})
    initial_ok=all(row['initial_match'] for row in pairs);prefix_ok=all(row['prefix_match'] for row in pairs)
    advantage=sum(mugs['live'].values())-sum(mugs['frozen'].values())
    median=float(np.median([r['frozen_minus_live_error_mm'] for r in reductions])) if reductions else None
    minimum_mugs=5 if split=='development' else 9
    minimum_complete=0 if split=='development' else 8
    count_advantage=2 if split=='development' else 3
    minimum_pairs=8 if split=='development' else 16
    requirements={
        'all_declared_cases':len(rows)==len(expected),'paired_initial_states':initial_ok,'paired_prefixes_before_correction':prefix_ok,
        'minimum_live_mug_success':all(mugs['live'][preset]>=minimum_mugs for preset in p['starting_presets']),
        'minimum_live_complete_workflows':all(counts['live'][preset]>=minimum_complete for preset in p['starting_presets']),
        'no_mug_regression_against_baseline':all(mugs['live'][preset]>=mugs['baseline'][preset] for preset in p['starting_presets']),
        'no_workflow_regression_against_baseline':all(counts['live'][preset]>=counts['baseline'][preset] for preset in p['starting_presets']),
        'no_mug_regression_against_frozen':all(mugs['live'][preset]>=mugs['frozen'][preset] for preset in p['starting_presets']),
        'live_feedback_benefit':advantage>=count_advantage or (len(reductions)>=minimum_pairs and median>=1.)}
    return {'schema':p['schema'],'split':split,'protocol_sha256':sha(protocol),'freeze_sha256':sha(root/'development-freeze.json'),
        'batch_sha256':sha(root/(split+'-batch.json')),'audit_source_sha256':sha(Path(__file__)),
        'planned':len(expected),'completed':len(rows),'trace_frames':frames,'portable_scenes_loaded':len(rows),
        'passed_workflows':counts,'passed_mugs':mugs,'correction_queries':query_total,'issued_corrections':issued,
        'maximum_motor_reproduction_difference':max_motor_difference,'live_minus_frozen_mug_successes':advantage,
        'mutually_successful_mug_pairs':len(reductions),'median_frozen_minus_live_mug_error_mm':median,
        'requirements':requirements,'gate_passed':all(requirements.values()),'pairs':pairs,'paired_error_reductions':reductions,'rows':rows,
        'scope':'All complete declared workflows retained, including early refusals. Paired-prefix mismatches fail promotion. Physical gate remains separate from model-interface accuracy.'}


def main(args):
    p=read(args.protocol);root=args.root or ROOT/p['raw_root']
    output=root/(args.split+'-audit.json')
    if output.exists():raise FileExistsError('Preserve the original paired audit.')
    result=audit(p,args.protocol,root,args.split)
    write(p,output,result)
    print({k:result[k] for k in ('split','completed','trace_frames','passed_workflows','passed_mugs','correction_queries','requirements','gate_passed')},flush=True)
    if args.freeze_final:
        if args.split!='development' or not result['gate_passed']:raise ValueError('Only passing complete development can authorize final scenes.')
        harness=ROOT/p.get('evaluation_harness','scripts/evaluate_mug_visual_correction.py')
        spec=importlib.util.spec_from_file_location('mug_promotion_harness',harness)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        inputs=module.fingerprint(p,args.protocol)
        if inputs!=read(root/'development-freeze.json')['inputs']:raise ValueError('Current implementation changed after development.')
        write(p,root/'evaluation-freeze.json',{'inputs':inputs,'development_gate_passed':True,'development_audit_sha256':sha(output),
            'development_freeze_sha256':sha(root/'development-freeze.json'),'evaluation_seeds':p['evaluation_seeds']})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',type=Path,default=ROOT/'docs/robotics/experiments/mug-visual-correction-v1.json')
    parser.add_argument('--root',type=Path)
    parser.add_argument('--split',choices=['development','evaluation'],required=True)
    parser.add_argument('--freeze-final',action='store_true')
    args=parser.parse_args();args.protocol=args.protocol.resolve();main(args)
