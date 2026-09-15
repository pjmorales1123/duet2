"""Package every paired spoon-release outcome and independently audit the evidence."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import mujoco
import numpy as np
from simulation_lab.storage import GIB, require_space

RAW = ROOT/'.run/spoon-release-v1'
MODEL = ROOT/'models/spoon_release_v1'
EVIDENCE = ROOT/'docs/robotics/evidence/spoon-release-v1'
PROTOCOL = ROOT/'docs/robotics/experiments/spoon-release-v1.json'


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rel(path):
    return path.resolve().relative_to(ROOT).as_posix()


def encoded(value):
    return (json.dumps(value,indent=2)+'\n').encode()


def evaluator():
    spec = importlib.util.spec_from_file_location('spoon_release_evaluator',ROOT/'scripts/evaluate_spoon_release.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def budget(expected):
    used = sum(p.stat().st_size for folder in (RAW,MODEL,EVIDENCE,ROOT/'training/spoon_release_v1')
               if folder.exists() for p in folder.rglob('*') if p.is_file())
    used += sum(p.stat().st_size for p in (ROOT/'.run/final-goal').glob('spoon-release-v1*') if p.is_file())
    if used+expected > read(PROTOCOL)['budget']['maximum_data_gib_including_packages']*GIB:
        raise ValueError('Raw and packaged evidence would exceed the declared budget.')
    return require_space(EVIDENCE,expected)


def physical_criteria(result):
    if result['status'] != 'succeeded':
        return
    m = result['metrics']
    if (not m['both_arms_parked'] or m['stable_release_and_park_s'] < .5-1e-8
            or m['unexpected_collisions'] or m['other_object_max_displacement_m'] > .004
            or m['parked_arm_max_motion_deg'] > 1 or result['teacher_updates'] != 0):
        raise ValueError('A passed skill contradicts the unchanged physical criteria.')
    if result['skill'] == 'drawer':
        if m['drawer_open_m'] < .105 or m['verified_handle_contact_s'] < .1:
            raise ValueError('The passed drawer lacks opening/contact evidence.')
    elif (m['placement_error_mm'] >= 8 or m['placement_z_error_mm'] >= 4 or m['hold_verified_s'] < 1.49
          or m['longest_unsupported_gap_s'] > .18 or m['speed_mm_s'] >= 3
          or m['placement_yaw_error_deg'] >= np.degrees(.175)):
        raise ValueError('A passed object placement contradicts its physical criteria.')
    details = result['policy_details']
    if (details.get('neural_runtime') != 'OpenVINO' or details.get('neural_execution_devices') != ['CPU']
            or details.get('inference_demonstration_actions') or details.get('simulator_clock_or_object_pose_input')):
        raise ValueError('A physical skill did not use the frozen OpenVINO CPU policy.')


def audit():
    module = evaluator()
    protocol = read(EVIDENCE/'protocol.json')
    if sha(EVIDENCE/'protocol.json') != sha(PROTOCOL):
        raise ValueError('The protocol changed.')
    training = read(MODEL/'training.json')
    if (training['selection']['sha256'] != sha(MODEL/'checkpoint/primitive.safetensors')
            or sha(MODEL/'openvino/primitive.safetensors') != training['selection']['sha256']):
        raise ValueError('Packaged weights differ from the sole fitted candidate.')
    export = read(MODEL/'openvino/benchmark.json')
    if not any(r['device']=='CPU' and r['parity_passed'] for r in export['devices']):
        raise ValueError('Candidate CPU parity is missing.')
    suite = read(MODEL/'suite.json')
    original = read(ROOT/protocol['baseline_suite'])
    for skill in original:
        actual = (MODEL/suite[skill]).resolve()
        expected = MODEL/'openvino' if skill=='spoon' else ((ROOT/protocol['baseline_suite']).parent/original[skill]).resolve()
        if actual != expected:
            raise ValueError('The portable suite changes another model.')
    totals = {'trials':0,'trace_frames':0,'portable_scenes_loaded':0,'paired_prefixes_verified':0,
              'maximum_final_spoon_metric_discrepancy_mm':0.,'splits':{}}
    for split in ('development','evaluation'):
        folder = EVIDENCE/split
        if not folder.exists():
            continue
        summary = read(folder/'summary.json')
        frozen_path = folder/'frozen-inputs.json'
        frozen = read(frozen_path)
        if frozen['split'] != split or frozen['inputs']['protocol_sha256'] != sha(PROTOCOL):
            raise ValueError('A physical freeze does not match the protocol.')
        for name,digest in frozen['inputs']['source_sha256'].items():
            if sha(EVIDENCE/'source'/name) != digest:
                raise ValueError('A frozen runtime source changed.')
        for name,digest in frozen['inputs']['asset_sha256'].items():
            if sha(ROOT/name) != digest:
                raise ValueError('A frozen simulation asset changed.')
        for controller,selection in frozen['inputs']['suites'].items():
            for name,digest in selection['files_sha256'].items():
                path = ROOT/name
                if controller=='candidate' and name.startswith('.run/spoon-release-v1/openvino/'):
                    path = MODEL/'openvino'/Path(name).name
                elif name=='.run/spoon-release-v1/candidate-suite.json':
                    path = EVIDENCE/'original-candidate-suite.json'
                if sha(path) != digest:
                    raise ValueError('An evaluated model/configuration changed: '+name)
        derived = module.aggregate(summary['rows'],split)
        for key,value in derived.items():
            if summary[key] != value:
                raise ValueError('The summary differs from independently recounted outcomes.')
        if not derived['complete'] or not derived['paired_initial_states_match']:
            raise ValueError('All declared paired outcomes and equal initial states are required.')
        for row in summary['rows']:
            base = folder/Path(row['report']).parent
            report_path = base/'report.json'
            r = read(report_path)
            if sha(report_path)!=row['report_sha256'] or r['freeze_sha256']!=sha(frozen_path):
                raise ValueError('A physical report differs from its frozen digest.')
            outcomes = r.get('task',{}).get('results',[])
            passed = r['status']=='succeeded' and r.get('task',{}).get('completed_steps')==module.STEPS[row['preset']]
            if row['passed']!=passed or r['physics_state_writes_during_control'] or r['hidden_forces']:
                raise ValueError('The summary or controller-assistance record is inconsistent.')
            for result in outcomes:
                physical_criteria(result)
            spoon = next((v for v in outcomes if v['skill']=='spoon'),None)
            if row['spoon_status'] != (None if spoon is None else spoon['status']):
                raise ValueError('A spoon outcome was miscounted.')
            with np.load(base/'states.npz',allow_pickle=False) as saved:
                trace = {k:saved[k] for k in saved.files}
            n = len(trace['time'])
            if any(len(v)!=n for v in trace.values()) or not n or np.any(np.diff(trace['time']) <= 0):
                raise ValueError('A complete monotonic synchronized trace is required.')
            if any(not np.isfinite(v).all() for k,v in trace.items() if k!='stage'):
                raise ValueError('A state trace is nonfinite.')
            model = mujoco.MjModel.from_xml_path(str((base/'scene.xml').resolve()))
            data = mujoco.MjData(model)
            initial = np.load(base/'initial-state.npy',allow_pickle=False)
            if hashlib.sha256(initial.astype(np.float64).tobytes()).hexdigest()!=r['initial_state_sha256']:
                raise ValueError('A retained initial state differs from its paired hash.')
            if len(initial)!=model.nq+model.nv+model.nu or trace['qpos'].shape[1]!=model.nq or trace['qvel'].shape[1]!=model.nv:
                raise ValueError('A scene and state have incompatible dimensions.')
            data.qpos[:] = trace['qpos'][-1]
            data.qvel[:] = trace['qvel'][-1]
            mujoco.mj_forward(model,data)
            if spoon:
                # The frozen dinner destination is encoded in its goal marker.
                destination = data.geom('spoon_place').xpos[:2]
                error = float(np.linalg.norm(data.body('spoon').xpos[:2]-destination)*1000)
                delta = abs(error-spoon['metrics']['placement_error_mm'])
                if delta > .001:
                    raise ValueError('Final spoon geometry contradicts its reported placement error.')
                totals['maximum_final_spoon_metric_discrepancy_mm'] = max(totals['maximum_final_spoon_metric_discrepancy_mm'],delta)
            totals['trials'] += 1
            totals['trace_frames'] += n
            totals['portable_scenes_loaded'] += 1
        for seed in frozen['seeds']:
            for preset in module.PRESETS:
                paths = [folder/f'{seed}-{preset}-{c}'/'states.npz' for c in module.CONTROLLERS]
                with np.load(paths[0],allow_pickle=False) as baseline, np.load(paths[1],allow_pickle=False) as candidate:
                    # The first spoon update changes targets, so exclude it.
                    masks = [saved['stage']!='spoon' for saved in (baseline,candidate)]
                    for key in ('time','qpos','qvel','stage','targets','progress'):
                        if not np.array_equal(baseline[key][masks[0]],candidate[key][masks[1]]):
                            raise ValueError('A paired workflow differs before the changed spoon skill.')
                totals['paired_prefixes_verified'] += 1
        totals['splits'][split] = {k:v for k,v in derived.items() if k not in ('rows','pairs')}
    if totals['splits']['development']['gate_passed'] and 'evaluation' not in totals['splits']:
        raise ValueError('A passing development candidate still needs every final paired trial.')
    totals['promoted'] = totals['splits'].get('evaluation',{}).get('gate_passed',False)
    totals['scope'] = 'Paired finite dinner workflows on AMD/RTX hardware; no broader workspace, Intel or human-voice claim.'
    return totals


def package():
    if MODEL.exists() or EVIDENCE.exists():
        raise FileExistsError('Both model and evidence package paths must be unused.')
    module = evaluator()
    development = read(RAW/'development/summary.json')
    splits = ['development']
    if not module.aggregate(development['rows'],'development')['complete']:
        raise ValueError('Wait for every development outcome.')
    if development['gate_passed']:
        final = read(RAW/'evaluation/summary.json')
        if not module.aggregate(final['rows'],'evaluation')['complete']:
            raise ValueError('Wait for every final outcome.')
        splits.append('evaluation')
    estimate = sum(p.stat().st_size for split in splits for p in (RAW/split).rglob('*') if p.is_file())+16*1024**2
    space = budget(estimate)
    MODEL.mkdir(parents=True)
    EVIDENCE.mkdir(parents=True)
    entries = []

    def put(target,payload,source=None,transformation='byte-identical copy'):
        budget(len(payload)+1024**2)
        target.parent.mkdir(parents=True,exist_ok=True)
        with target.open('xb') as stream:
            stream.write(payload)
        entries.append({'file':rel(target),'sha256':sha(target),'bytes':len(payload),
                        'source':None if source is None else rel(source),'source_sha256':None if source is None else sha(source),
                        'transformation':transformation})

    for folder in ('checkpoint','openvino'):
        source = RAW/('fit/step-012000' if folder=='checkpoint' else 'openvino')
        for path in sorted(source.iterdir()):
            if path.is_file():
                put(MODEL/folder/path.name,path.read_bytes(),path)
    put(MODEL/'training.json',(RAW/'fit/training.json').read_bytes(),RAW/'fit/training.json')
    original = read(ROOT/read(PROTOCOL)['baseline_suite'])
    parent = (ROOT/read(PROTOCOL)['baseline_suite']).parent
    suite = {skill:os.path.relpath((parent/value).resolve(),MODEL).replace('\\','/') for skill,value in original.items()}
    suite['spoon'] = 'openvino'
    put(MODEL/'suite.json',encoded(suite),transformation='Portable paths; only spoon checkpoint replaced.')
    put(EVIDENCE/'protocol.json',PROTOCOL.read_bytes(),PROTOCOL)
    put(EVIDENCE/'original-candidate-suite.json',(RAW/'candidate-suite.json').read_bytes(),RAW/'candidate-suite.json')
    for split in splits:
        source = RAW/split
        for path in sorted(source.rglob('*')):
            if not path.is_file():
                continue
            name = path.relative_to(source)
            if name.parts[0]=='source':
                target = EVIDENCE/name
                if target.exists():
                    if sha(target)!=sha(path):raise ValueError('The two physical source snapshots differ.')
                    continue
            else:
                target = EVIDENCE/split/name
            payload,change = path.read_bytes(),'byte-identical copy'
            if path.name=='scene.xml':
                tree = ET.fromstring(payload)
                compiler = tree.find('compiler')
                assets = (path.parent/compiler.get('meshdir')).resolve()
                if not assets.is_relative_to(ROOT/'simulation_lab/assets'):
                    raise ValueError('Unexpected scene mesh directory.')
                compiler.set('meshdir',os.path.relpath(assets,target.parent).replace('\\','/'))
                payload,change = ET.tostring(tree,encoding='utf-8'),'Only meshdir remapped to the same repository assets.'
            elif path.suffix=='.log':
                for prefix in (str(ROOT),ROOT.as_posix()):
                    payload = payload.replace(prefix.encode(),b'{repository}')
                if payload!=path.read_bytes():change='Local repository prefix redacted from console paths.'
            put(target,payload,path,change)
    for name in ('train_spoon_release.py','export_primitive_openvino.py','package_spoon_release.py'):
        path = ROOT/'scripts'/name
        put(EVIDENCE/'source/scripts'/name,path.read_bytes(),path)
    result = audit()
    result['disk_preflight'] = {'free_bytes':space['free_bytes'],'expected_package_bytes':estimate,'reserve_bytes':space['reserve_bytes']}
    put(EVIDENCE/'audit.json',encoded(result),transformation='Independent hash, outcome, physical metric and trace audit.')
    manifest = {'files':entries,'originals_modified':False,'all_declared_outcomes_retained':True}
    budget(1024**2)
    with (EVIDENCE/'package-manifest.json').open('xb') as stream:
        stream.write(encoded(manifest))
    return result


def restore_evaluation_inputs():
    """Restore exact frozen lookup paths on a fresh clone without another fit."""
    if RAW.exists():
        raise FileExistsError('The frozen input destination already exists; preserve it.')
    copies = [(MODEL/'training.json',RAW/'fit/training.json'),
              (EVIDENCE/'original-candidate-suite.json',RAW/'candidate-suite.json')]
    copies += [(p,RAW/'openvino'/p.name) for p in sorted((MODEL/'openvino').iterdir()) if p.is_file()]
    budget(sum(p.stat().st_size for p,_ in copies)+1024**2)
    for source,target in copies:
        require_space(target,source.stat().st_size+1024**2)
        target.parent.mkdir(parents=True,exist_ok=True)
        with target.open('xb') as stream:
            stream.write(source.read_bytes())
    module = evaluator()
    frozen = read(EVIDENCE/'development/frozen-inputs.json')
    module.validate(frozen)
    return {'restored_files':len(copies),'destination':rel(RAW),'frozen_inputs_match':True}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--verify-only',action='store_true')
    mode.add_argument('--restore-evaluation-inputs',action='store_true')
    args = parser.parse_args()
    if args.verify_only or args.restore_evaluation_inputs:
        for entry in read(EVIDENCE/'package-manifest.json')['files']:
            if sha(ROOT/entry['file'])!=entry['sha256']:
                raise ValueError('A packaged file changed: '+entry['file'])
        result = audit()
        saved = read(EVIDENCE/'audit.json')
        if any(saved[key]!=value for key,value in result.items()):
            raise ValueError('The independent audit differs from the stored result.')
        if args.restore_evaluation_inputs:
            result = restore_evaluation_inputs()
    else:
        result = package()
    print(json.dumps(result))
