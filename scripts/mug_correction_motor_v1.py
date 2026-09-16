"""Bounded offline data, fit, export and fresh scoring for local mug corrections."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import mujoco
import numpy as np
import torch
from safetensors.torch import load_file, save_file
from simulation_lab.mug_correction_motor import MugDifferentialMotorNet, MugRobotGeometry, assess_delta
from simulation_lab.storage import require_space


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def space(p, destination, expected):
    roots = [ROOT/p[key] for key in ('raw_root', 'model_package', 'training_package', 'evidence_package')]
    used = sum(f.stat().st_size for root in roots if root.exists() for f in root.rglob('*') if f.is_file())
    if used+expected > p['budget']['maximum_combined_gib']*1024**3:
        raise ValueError('Declared motor data budget exceeded.')
    return require_space(destination, expected, p['budget']['reserve_gib']*1024**3)


def write(p, path, data):
    payload = (json.dumps(data, indent=2)+'\n').encode('utf-8')
    space(p, path, len(payload)+1024)
    with Path(path).open('xb') as stream:
        stream.write(payload)


def sources(p, protocol_path):
    paths = [protocol_path, Path(__file__), ROOT/'simulation_lab/mug_correction_motor.py', ROOT/'simulation_lab/storage.py',
             ROOT/'scripts/audit_mug_correction_motor.py', ROOT/'scripts/torch_robot_geometry.py', ROOT/p['scene'], ROOT/p['training_source']]
    # The immutable episode manifest covers the geometry/assets already in Git.
    inherited = read(ROOT/'training/mug_keypoint_observer_v1/training/manifest.json')['source_sha256']
    result = dict(inherited)
    for path in paths:
        result[path.relative_to(ROOT).as_posix()] = sha(path)
    for name, digest in result.items():
        if sha(ROOT/name) != digest:
            raise ValueError('A declared geometry/input dependency changed: '+name)
    return result


def aggregate(rows, p):
    errors = [r['position_error_mm'] for r in rows]
    axes = [r['axis_error'] for r in rows]
    result = {'attempted': len(rows), 'accepted': sum(r['accepted'] for r in rows),
        'p95_position_error_mm': float(np.quantile(errors, .95)), 'maximum_position_error_mm': max(errors), 'maximum_axis_error': max(axes)}
    gate = p['gate']
    result['gate_passed'] = bool(result['accepted']/len(rows) >= gate['minimum_accepted_fraction']
        and result['p95_position_error_mm'] <= gate['maximum_p95_position_error_mm']
        and max(errors) <= gate['maximum_position_error_mm'] and max(axes) <= gate['maximum_axis_error'])
    return result


def data_gate(p, folder, protocol_path):
    manifest = read(folder/'manifest.json')
    if manifest['protocol_sha256'] != sha(protocol_path) or sha(folder/'data.npz') != manifest['data_sha256']:
        raise ValueError('A motor dataset changed.')
    for name, digest in manifest['source_sha256'].items():
        if sha(ROOT/name) != digest:
            raise ValueError('A frozen data dependency changed: '+name)
    with np.load(folder/'data.npz', allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    return manifest, arrays


def collect(p, args):
    raw = ROOT/p['raw_root']
    folder = raw/args.split
    if folder.exists():
        raise FileExistsError('Preserve earlier datasets.')
    if args.split == 'evaluation' and not read(raw/'openvino/parity.json')['passed']:
        raise ValueError('Fresh kinematics require successful frozen export first.')
    preflight = space(p, folder, 32*1024**2)
    frozen = sources(p, args.protocol)
    model = mujoco.MjModel.from_xml_path(str(ROOT/p['scene']))
    geometry = MugRobotGeometry(model)
    with np.load(ROOT/p['training_source'], allow_pickle=False) as archive:
        bounds, seconds, actions = (archive[name] for name in ('bounds', 'seconds', 'actions'))
    count = p['data'][args.split+'_states']
    rng = np.random.default_rng(p['data'][args.split+'_seed'])
    episode = rng.integers(0, len(bounds), count)
    phase = rng.uniform(*p['data']['nominal_seconds_range'], count)
    jitter = rng.uniform(-p['data']['joint_jitter_half_range_rad'], p['data']['joint_jitter_half_range_rad'], (count, 5))
    translation = np.column_stack((rng.uniform(-1, 1, (count, 2)), np.zeros(count)))
    translation = (translation/np.maximum(1, np.linalg.norm(translation, axis=1))[:, None]*p['data']['maximum_translation_norm_m']).astype(np.float32)
    joints = np.zeros((count, 5), np.float32)
    coefficients, labels = np.zeros((count, 5, 3), np.float32), np.zeros((count, 5), np.float32)
    accepted, rows = np.zeros(count, bool), []
    jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    lo, hi = model.actuator_ctrlrange[6:11].T
    started = time.perf_counter()
    folder.mkdir(parents=True)
    for index in range(count):
        a, b = bounds[episode[index]]
        q = [np.interp(phase[index], seconds[a:b], actions[a:b, j]) for j in range(6, 11)]
        joints[index] = np.clip(q+jitter[index], lo+.005, hi-.005)
        point, rotation = geometry.pose(joints[index])
        mujoco.mj_comPos(model, geometry.scratch)
        mujoco.mj_jac(model, geometry.scratch, jp, jr, point, geometry.body)
        axis = rotation[:, p['axis_local_index']]
        x, y, z = axis
        skew = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
        jac = np.vstack((jp[:, 6:11], -p['data']['jacobian_axis_weight_m']*skew@jr[:, 6:11]))
        matrix = np.linalg.solve(jac.T@jac+np.eye(5)*p['data']['damping'], jac.T[:, :3])
        coefficients[index] = matrix
        labels[index] = joints[index]+coefficients[index]@translation[index]
        row = assess_delta(geometry, joints[index], translation[index], labels[index]-joints[index], {'minimum':lo,'maximum':hi},
            {**p['runtime_guard'], 'maximum_position_error_mm':p['data']['maximum_label_position_error_mm'], 'maximum_axis_error':p['data']['maximum_label_axis_error']})
        row.update(index=index, episode_index=int(episode[index]), nominal_seconds=float(phase[index]))
        accepted[index] = row['accepted']
        rows.append(row)
        if index % 512 == 0:
            space(p, folder, 16*1024**2)
            print({'split':args.split,'attempted':index+1,'accepted':int(accepted[:index+1].sum())},flush=True)
            if time.perf_counter()-started > p['budget']['maximum_generation_minutes_per_split']*60:
                raise TimeoutError('Declared kinematic generation budget reached.')
    space(p, folder, 32*1024**2)
    with (folder/'data.npz').open('xb') as stream:
        np.savez_compressed(stream,joints=joints,translation=translation,coefficients=coefficients,labels=labels,
            accepted=accepted,episode_index=episode,nominal_seconds=phase,joint_jitter=jitter)
    manifest = {'schema':p['schema'],'split':args.split,'protocol_sha256':sha(args.protocol),'source_sha256':frozen,
        'states':count,'accepted_labels':int(accepted.sum()),'label_coverage_passed':float(accepted.mean())>=p['data']['minimum_label_coverage'],
        'data_sha256':sha(folder/'data.npz'),'rows':rows,'preflight':preflight,'wall_seconds':time.perf_counter()-started,
        'parent_git_revision':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'scope':'Offline local differential kinematics only; no contact-physics episode or learned control.'}
    write(p,folder/'manifest.json',manifest)
    print({k:manifest[k] for k in ('split','states','accepted_labels','label_coverage_passed','wall_seconds')},flush=True)


def score(p, data, predict, support):
    model = mujoco.MjModel.from_xml_path(str(ROOT/p['scene']))
    geometry = MugRobotGeometry(model)
    predictions = predict(data['joints'], data['translation'])
    rows = []
    for i, (q, shift, delta) in enumerate(zip(data['joints'], data['translation'], predictions)):
        row = assess_delta(geometry,q,shift,delta,support,p['runtime_guard'])
        row.update(index=i, offline_label_accepted=bool(data['accepted'][i]))
        rows.append(row)
    return {'summary':aggregate(rows,p),'rows':rows}


def torch_predict(network, device):
    def predict(q, d):
        with torch.inference_mode():
            return network(torch.from_numpy(q).to(device),torch.from_numpy(d).to(device)).cpu().numpy()
    return predict


def train(p, args):
    raw, folder = ROOT/p['raw_root'], ROOT/p['raw_root']/'fit'
    if folder.exists() or (raw/'evaluation').exists():
        raise FileExistsError('Preserve fits; never fit after exposing fresh data.')
    manifests, arrays = {}, {}
    for split in ('training','development'):
        manifests[split], arrays[split] = data_gate(p,raw/split,args.protocol)
        if not manifests[split]['label_coverage_passed']:
            raise ValueError('Both initial label coverage gates must pass before fitting.')
        audit=read(raw/split/'audit.json')
        if not audit['passed'] or audit['data_sha256']!=sha(raw/split/'data.npz'):
            raise ValueError('Both independent geometry audits must pass before fitting.')
    if not torch.cuda.is_available():
        raise RuntimeError('The declared fit requires CUDA.')
    torch.set_num_threads(2)
    torch.manual_seed(p['training']['seed'])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.reset_peak_memory_stats()
    preflight = space(p,folder,32*1024**2)
    folder.mkdir()
    inputs = arrays['training']
    q = torch.from_numpy(inputs['joints'][inputs['accepted']]).cuda()
    targets = torch.from_numpy(inputs['coefficients'][inputs['accepted']]).cuda()
    network = MugDifferentialMotorNet().cuda()
    with torch.no_grad():
        network.center.copy_(q.mean(0));network.scale.copy_(q.std(0).clamp_min(.05))
        network.minimum.copy_(q.amin(0)-p['runtime_guard']['joint_support_margin_rad'])
        network.maximum.copy_(q.amax(0)+p['runtime_guard']['joint_support_margin_rad'])
    support={k:getattr(network,k).cpu().numpy().tolist() for k in ('minimum','maximum')}
    settings=p['training']
    optimizer=torch.optim.AdamW(network.parameters(),lr=settings['learning_rate'],weight_decay=settings['weight_decay'])
    started,curve,candidates=time.perf_counter(),[],[]
    for step in range(1,settings['updates']+1):
        if time.perf_counter()-started > p['budget']['maximum_fit_minutes']*60 or torch.cuda.max_memory_reserved()/1024**3 > p['budget']['maximum_torch_reserved_gib']:
            raise RuntimeError('Declared fit resource budget exceeded.')
        fraction=(step-1)/(settings['updates']-1)
        lr=settings['final_learning_rate']+(settings['learning_rate']-settings['final_learning_rate'])*(1+math.cos(math.pi*fraction))/2
        for group in optimizer.param_groups:group['lr']=lr
        ids=torch.randint(len(q),(settings['batch_size'],),device='cuda')
        loss=((network.coefficients(q[ids])-targets[ids])/p['network']['coefficient_scale_rad_per_m']).square().mean()
        if not torch.isfinite(loss):raise ValueError('Nonfinite differential motor loss.')
        optimizer.zero_grad(set_to_none=True);loss.backward()
        torch.nn.utils.clip_grad_norm_(network.parameters(),settings['gradient_clip_norm']);optimizer.step()
        if step==1 or step%500==0:
            torch.cuda.synchronize();space(p,folder,8*1024**2)
            row={'step':step,'loss':float(loss.detach()),'wall_seconds':time.perf_counter()-started,'peak_torch_reserved_gib':torch.cuda.max_memory_reserved()/1024**3}
            curve.append(row);print(row,flush=True)
        if step in settings['checkpoints']:
            out=folder/f'step-{step:06d}'
            space(p,out,16*1024**2);out.mkdir()
            save_file({k:v.detach().cpu().contiguous() for k,v in network.state_dict().items()},str(out/'model.safetensors'))
            result=score(p,arrays['development'],torch_predict(network,'cuda'),support)
            result.update(checkpoint_sha256=sha(out/'model.safetensors'),protocol_sha256=sha(args.protocol))
            write(p,out/'development.json',result)
            candidates.append({'step':step,'checkpoint':(out/'model.safetensors').relative_to(ROOT).as_posix(),
                'checkpoint_sha256':result['checkpoint_sha256'],'development':result['summary']})
            print(candidates[-1],flush=True)
    qualified=[c for c in candidates if c['development']['gate_passed']]
    selected=min(qualified,key=lambda c:(c['development']['p95_position_error_mm'],c['development']['maximum_position_error_mm'],c['step'])) if qualified else None
    report={'schema':p['schema'],'protocol_sha256':sha(args.protocol),'source_sha256':sources(p,args.protocol),
        'dataset_sha256':{s:sha(raw/s/'data.npz') for s in arrays},'support':support,'completed_updates':settings['updates'],
        'curve':curve,'candidates':candidates,'selected':selected,'development_gate_passed':bool(selected),'preflight':preflight,
        'wall_seconds':time.perf_counter()-started,'peak_torch_reserved_gib':torch.cuda.max_memory_reserved()/1024**3,
        'gpu':torch.cuda.get_device_name(0),'torch':str(torch.__version__),'cuda':str(torch.version.cuda),
        'parent_git_revision':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'new_physical_trials':0}
    write(p,folder/'training.json',report)
    print({'development_gate_passed':bool(selected),'selected':selected,'wall_seconds':report['wall_seconds']},flush=True)


def selected_model(p,args):
    raw=ROOT/p['raw_root']
    fit=read(raw/'fit/training.json')
    if not fit['development_gate_passed'] or fit['protocol_sha256']!=sha(args.protocol):
        raise ValueError('Successful frozen development selection is required.')
    for name,digest in fit['source_sha256'].items():
        if sha(ROOT/name)!=digest:raise ValueError('A frozen motor source changed: '+name)
    checkpoint=ROOT/fit['selected']['checkpoint']
    if sha(checkpoint)!=fit['selected']['checkpoint_sha256']:raise ValueError('Selected motor weights changed.')
    return fit,checkpoint


def export_or_evaluate(p,args):
    import openvino as ov
    raw=ROOT/p['raw_root']
    exporting=args.mode=='export'
    output=raw/('openvino/parity.json' if exporting else 'evaluation/evaluation.json')
    if output.exists() or (exporting and output.parent.exists()):raise FileExistsError('Preserve every export/evaluation.')
    fit,checkpoint=selected_model(p,args)
    preflight=space(p,output,32*1024**2)
    torch.set_num_threads(2)
    network=MugDifferentialMotorNet().eval();network.load_state_dict(load_file(str(checkpoint)))
    if exporting:
        sample=(torch.zeros(1,5),torch.zeros(1,3))
        converted=ov.convert_model(torch.jit.trace(network,sample),example_input=sample)
        output.parent.mkdir()
        ov.save_model(converted,output.parent/'motor.xml',compress_to_fp16=False)
    else:
        parity=read(raw/'openvino/parity.json')
        if not parity['passed'] or parity['checkpoint_sha256']!=sha(checkpoint):raise ValueError('Fresh scoring requires qualified export.')
        for name,digest in parity['ir_sha256'].items():
            if sha(raw/'openvino'/name)!=digest:raise ValueError('Evaluated IR changed.')
    core=ov.Core()
    compiled=core.compile_model(str(raw/'openvino/motor.xml'),'CPU',{'PERFORMANCE_HINT':'LATENCY','INFERENCE_PRECISION_HINT':'f32','INFERENCE_NUM_THREADS':2})
    manifest,data=data_gate(p,raw/('development' if exporting else 'evaluation'),args.protocol)
    audited=raw/('development' if exporting else 'evaluation')
    audit=read(audited/'audit.json')
    if not audit['passed'] or audit['data_sha256']!=sha(audited/'data.npz'):
        raise ValueError('Independent geometry audit required before export/fresh scoring.')
    latency=[]
    def predict(q,d):
        values=[]
        for joint,delta in zip(q,d):
            start=time.perf_counter();values.append(compiled([joint[None],delta[None]])[0].copy()[0]);latency.append((time.perf_counter()-start)*1000)
        return np.asarray(values)
    actual=predict(data['joints'],data['translation'])
    reference=torch_predict(network,'cpu')(data['joints'],data['translation'])
    parity_errors=np.max(abs(actual-reference),axis=1)
    result=score(p,data,lambda q,d:actual,fit['support'])
    result.update(schema=p['schema'],protocol_sha256=sha(args.protocol),checkpoint_sha256=sha(checkpoint),
        ir_sha256={name:sha(raw/'openvino'/name) for name in ('motor.xml','motor.bin')},source_sha256=sha(Path(__file__)),
        maximum_joint_delta_parity_rad=float(parity_errors.max()),all_parity_rad=parity_errors.tolist(),
        median_inference_ms=float(np.median(latency)),all_inference_ms=latency,device=core.get_property('CPU','FULL_DEVICE_NAME'),
        precision=str(compiled.get_property('INFERENCE_PRECISION_HINT')),preflight=preflight,
        passed=bool(result['summary']['gate_passed'] and parity_errors.max()<=p['export']['maximum_joint_delta_parity_rad']),new_physical_trials=0,
        scope='Offline CPU local motor geometry only; no contact, camera-control, Intel or task-success claim.')
    write(p,output,result)
    print({k:result[k] for k in ('passed','summary','maximum_joint_delta_parity_rad','median_inference_ms','device')},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',type=Path,default=ROOT/'docs/robotics/experiments/mug-correction-motor-v1.json')
    parser.add_argument('--mode',choices=['collect','train','export','evaluate'],required=True)
    parser.add_argument('--split',choices=['training','development','evaluation'])
    args=parser.parse_args()
    p=read(args.protocol)
    if sha(ROOT/p['training_source'])!=p['training_source_sha256']:raise ValueError('The preserved mug training input changed.')
    if args.mode=='collect':
        if not args.split:parser.error('--split is required for collection')
        collect(p,args)
    elif args.mode=='train':train(p,args)
    else:export_or_evaluate(p,args)
