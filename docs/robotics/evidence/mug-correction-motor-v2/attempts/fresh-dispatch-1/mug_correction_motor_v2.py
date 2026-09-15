"""One separately bounded motor fit with joint-limited Cartesian steps."""
import argparse
import math
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import mujoco
import numpy as np
import torch
from safetensors.torch import load_file,save_file
from scripts.mug_correction_motor_v1 import sha,read,space,write,data_gate,sources as original_sources,aggregate
from simulation_lab.mug_correction_motor import MugDifferentialMotorNet,MugRobotGeometry
from simulation_lab.mug_correction_limits import limited_delta,assess_limited


def sources(p,protocol):
    result=original_sources(p,protocol)
    for path in (Path(__file__),ROOT/'simulation_lab/mug_correction_limits.py',ROOT/p['prior_protocol']):
        result[path.relative_to(ROOT).as_posix()]=sha(path)
    for item in p['inherited_data'].values():
        if sha(ROOT/item['path'])!=item['sha256']:raise ValueError('A retained prior numeric input changed.')
        result[item['path']]=item['sha256']
    if sha(ROOT/p['prior_protocol'])!=p['prior_protocol_sha256']:raise ValueError('The stopped prior protocol changed.')
    return result


def fresh_inputs(p,model):
    """Same declared sampling distribution, newly seeded; offline scratch only."""
    rng=np.random.default_rng(p['data']['evaluation_seed'])
    count=p['data']['evaluation_states']
    with np.load(ROOT/p['training_source'],allow_pickle=False) as archive:
        bounds,seconds,actions=(archive[name] for name in ('bounds','seconds','actions'))
    episode=rng.integers(0,len(bounds),count)
    phase=rng.uniform(*p['data']['nominal_seconds_range'],count)
    jitter=rng.uniform(-p['data']['joint_jitter_half_range_rad'],p['data']['joint_jitter_half_range_rad'],(count,5))
    shift=np.column_stack((rng.uniform(-1,1,(count,2)),np.zeros(count)))
    shift=(shift/np.maximum(1,np.linalg.norm(shift,axis=1))[:,None]*p['data']['maximum_translation_norm_m']).astype(np.float32)
    q=np.zeros((count,5),np.float32);matrix=np.zeros((count,5,3),np.float32)
    geometry=MugRobotGeometry(model)
    jp,jr=np.zeros((3,model.nv)),np.zeros((3,model.nv))
    lo,hi=model.actuator_ctrlrange[6:11].T
    for i in range(count):
        a,b=bounds[episode[i]]
        base=[np.interp(phase[i],seconds[a:b],actions[a:b,j]) for j in range(6,11)]
        q[i]=np.clip(base+jitter[i],lo+.005,hi-.005)
        point,rotation=geometry.pose(q[i]);mujoco.mj_comPos(model,geometry.scratch)
        mujoco.mj_jac(model,geometry.scratch,jp,jr,point,geometry.body)
        x,y,z=rotation[:,2]
        skew=np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])
        jac=np.vstack((jp[:,6:11],-p['data']['jacobian_axis_weight_m']*skew@jr[:,6:11]))
        matrix[i]=np.linalg.solve(jac.T@jac+np.eye(5)*p['data']['damping'],jac.T[:,:3])
    return {'joints':q,'translation':shift,'coefficients':matrix,'episode_index':episode,'nominal_seconds':phase,'joint_jitter':jitter}


def prepare(p,args):
    raw,folder=ROOT/p['raw_root'],ROOT/p['raw_root']/args.split
    if folder.exists():raise FileExistsError('Preserve existing batches.')
    if args.split=='evaluation':
        fit,checkpoint=frozen(p,args)
        parity=read(raw/'openvino/parity.json')
        if not parity['passed'] or parity['protocol_sha256']!=sha(args.protocol) or parity['checkpoint_sha256']!=sha(checkpoint):
            raise ValueError('Fresh inputs require a frozen passing export.')
        for name,digest in parity['ir_sha256'].items():
            if sha(raw/'openvino'/name)!=digest:raise ValueError('Frozen motor IR changed.')
    frozen=sources(p,args.protocol)
    preflight=space(p,folder,32*1024**2)
    model=mujoco.MjModel.from_xml_path(str(ROOT/p['scene']))
    geometry=MugRobotGeometry(model)
    started=time.perf_counter()
    if args.split=='evaluation':
        data=fresh_inputs(p,model);origin=None
    else:
        item=p['inherited_data'][args.split];origin=item.copy()
        with np.load(ROOT/item['path'],allow_pickle=False) as archive:
            data={name:archive[name].copy() for name in archive.files if name not in ('accepted','labels')}
    count=p['data'][args.split+'_states']
    if len(data['joints'])!=count:raise ValueError('Every declared position must remain.')
    original=data['translation'].copy()
    data['requested_translation']=original
    data['translation']=np.zeros((count,3),float)
    data['labels']=np.zeros((count,5),float)
    data['accepted']=np.zeros(count,bool);data['step_fraction']=np.zeros(count,float)
    lo,hi=model.actuator_ctrlrange[6:11].T
    rows=[]
    for i,(q,matrix,requested) in enumerate(zip(data['joints'],data['coefficients'],original)):
        raw_delta=matrix@requested
        delta,effective,fraction=limited_delta(raw_delta,requested,p['step_sizing']['maximum_joint_delta_rad'])
        data['labels'][i]=q.astype(float)+delta
        data['translation'][i]=effective;data['step_fraction'][i]=fraction
        # Match exactly the endpoint subtraction used by the independent audit.
        from simulation_lab.mug_correction_motor import assess_delta
        row=assess_delta(geometry,q,effective,data['labels'][i]-q,{'minimum':lo,'maximum':hi},
            {**p['runtime_guard'],'maximum_position_error_mm':p['data']['maximum_label_position_error_mm']})
        # Endpoint arithmetic can add a last-bit roundoff to a capped joint.
        # The authoritative command is delta, assessed independently below.
        command=assess_limited(geometry,q,requested,raw_delta,{'minimum':lo,'maximum':hi},
            {**p['runtime_guard'],'maximum_position_error_mm':p['data']['maximum_label_position_error_mm']},p['step_sizing'])
        row.update(index=i,step_fraction=fraction,original_requested_translation_m=requested.tolist(),
                   accepted=command['accepted'],episode_index=int(data['episode_index'][i]),nominal_seconds=float(data['nominal_seconds'][i]))
        data['accepted'][i]=row['accepted'];rows.append(row)
        if i%512==0:
            space(p,folder,16*1024**2)
            if time.perf_counter()-started>p['budget']['maximum_generation_minutes_per_split']*60:raise TimeoutError('Data time budget exceeded.')
    space(p,folder,32*1024**2);folder.mkdir(parents=True)
    with (folder/'data.npz').open('xb') as stream:np.savez_compressed(stream,**data)
    result={'schema':p['schema'],'protocol_sha256':sha(args.protocol),'source_sha256':frozen,'origin':origin,'split':args.split,
        'states':count,'accepted_labels':int(data['accepted'].sum()),'label_coverage_passed':float(data['accepted'].mean())>=p['data']['minimum_label_coverage'],
        'step_fraction':{'minimum':float(data['step_fraction'].min()),'median':float(np.median(data['step_fraction']))},
        'data_sha256':sha(folder/'data.npz'),'rows':rows,'preflight':preflight,'wall_seconds':time.perf_counter()-started,
        'parent_git_revision':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'scope':'Every original configuration retained with smaller joint-limited labels; offline kinematics only.'}
    write(p,folder/'manifest.json',result)
    print({k:result[k] for k in ('split','states','accepted_labels','label_coverage_passed','step_fraction','wall_seconds')},flush=True)


def checked(p,args,split):
    folder=ROOT/p['raw_root']/split
    manifest,data=data_gate(p,folder,args.protocol)
    audit=read(folder/'audit.json')
    if not audit['passed'] or audit['data_sha256']!=sha(folder/'data.npz'):raise ValueError('Independent geometry audit required.')
    return manifest,data


def score(p,data,predictions,support):
    model=mujoco.MjModel.from_xml_path(str(ROOT/p['scene']));geometry=MugRobotGeometry(model)
    rows=[]
    for i,(q,requested,delta) in enumerate(zip(data['joints'],data['requested_translation'],predictions)):
        row=assess_limited(geometry,q,requested,delta,support,p['runtime_guard'],p['step_sizing'])
        row.update(index=i,offline_label_accepted=bool(data['accepted'][i]));rows.append(row)
    summary=aggregate(rows,p)
    fractions=[r['step_fraction'] for r in rows]
    summary.update(minimum_step_fraction=min(fractions),median_step_fraction=float(np.median(fractions)))
    summary['gate_passed']=bool(summary['gate_passed'] and min(fractions)>=p['step_sizing']['minimum_fraction'])
    return {'summary':summary,'rows':rows}


def predict_torch(network,data,device):
    with torch.inference_mode():
        return network(torch.from_numpy(data['joints']).to(device),torch.from_numpy(data['requested_translation']).to(device)).cpu().numpy()


def train(p,args):
    raw,folder=ROOT/p['raw_root'],ROOT/p['raw_root']/'fit'
    if folder.exists() or (raw/'evaluation').exists():raise FileExistsError('Preserve fits; do not fit after fresh evaluation.')
    arrays={}
    for split in ('training','development'):
        manifest,arrays[split]=checked(p,args,split)
        if not manifest['label_coverage_passed']:raise ValueError('Initial label coverage failed; no fit.')
    if not torch.cuda.is_available():raise RuntimeError('The declared fit requires CUDA.')
    torch.set_num_threads(2);torch.manual_seed(p['training']['seed'])
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.cuda.reset_peak_memory_stats()
    preflight=space(p,folder,32*1024**2);folder.mkdir()
    data=arrays['training'];q=torch.from_numpy(data['joints'][data['accepted']]).cuda()
    labels=torch.from_numpy(data['coefficients'][data['accepted']]).cuda()
    network=MugDifferentialMotorNet().cuda()
    with torch.no_grad():
        network.center.copy_(q.mean(0));network.scale.copy_(q.std(0).clamp_min(.05))
        network.minimum.copy_(q.amin(0)-p['runtime_guard']['joint_support_margin_rad'])
        network.maximum.copy_(q.amax(0)+p['runtime_guard']['joint_support_margin_rad'])
    support={k:getattr(network,k).cpu().numpy().tolist() for k in ('minimum','maximum')}
    settings=p['training'];optimizer=torch.optim.AdamW(network.parameters(),lr=settings['learning_rate'],weight_decay=settings['weight_decay'])
    started,curve,candidates=time.perf_counter(),[],[]
    for step in range(1,settings['updates']+1):
        if time.perf_counter()-started>p['budget']['maximum_fit_minutes']*60 or torch.cuda.max_memory_reserved()/1024**3>p['budget']['maximum_torch_reserved_gib']:
            raise RuntimeError('Declared fit budget exceeded.')
        fraction=(step-1)/(settings['updates']-1)
        lr=settings['final_learning_rate']+(settings['learning_rate']-settings['final_learning_rate'])*(1+math.cos(math.pi*fraction))/2
        for group in optimizer.param_groups:group['lr']=lr
        ids=torch.randint(len(q),(settings['batch_size'],),device='cuda')
        loss=((network.coefficients(q[ids])-labels[ids])/p['network']['coefficient_scale_rad_per_m']).square().mean()
        if not torch.isfinite(loss):raise ValueError('Nonfinite motor loss.')
        optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(network.parameters(),settings['gradient_clip_norm']);optimizer.step()
        if step==1 or step%500==0:
            torch.cuda.synchronize();space(p,folder,8*1024**2)
            row={'step':step,'loss':float(loss.detach()),'wall_seconds':time.perf_counter()-started,'peak_torch_reserved_gib':torch.cuda.max_memory_reserved()/1024**3}
            curve.append(row);print(row,flush=True)
        if step in settings['checkpoints']:
            out=folder/f'step-{step:06d}';space(p,out,16*1024**2);out.mkdir()
            checkpoint=out/'model.safetensors';save_file({k:v.detach().cpu().contiguous() for k,v in network.state_dict().items()},str(checkpoint))
            result=score(p,arrays['development'],predict_torch(network,arrays['development'],'cuda'),support)
            result.update(checkpoint_sha256=sha(checkpoint),protocol_sha256=sha(args.protocol))
            write(p,out/'development.json',result)
            candidate={'step':step,'checkpoint':checkpoint.relative_to(ROOT).as_posix(),'checkpoint_sha256':sha(checkpoint),'development':result['summary']}
            candidates.append(candidate);print(candidate,flush=True)
    qualified=[r for r in candidates if r['development']['gate_passed']]
    selected=min(qualified,key=lambda r:(r['development']['p95_position_error_mm'],r['development']['maximum_position_error_mm'],r['step'])) if qualified else None
    result={'schema':p['schema'],'protocol_sha256':sha(args.protocol),'source_sha256':sources(p,args.protocol),
        'dataset_sha256':{s:sha(raw/s/'data.npz') for s in arrays},'support':support,'completed_updates':settings['updates'],
        'curve':curve,'candidates':candidates,'selected':selected,'development_gate_passed':bool(selected),'preflight':preflight,
        'wall_seconds':time.perf_counter()-started,'peak_torch_reserved_gib':torch.cuda.max_memory_reserved()/1024**3,
        'gpu':torch.cuda.get_device_name(0),'torch':str(torch.__version__),'cuda':str(torch.version.cuda),
        'parent_git_revision':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'new_physical_trials':0}
    write(p,folder/'training.json',result)
    print({'development_gate_passed':bool(selected),'selected':selected,'wall_seconds':result['wall_seconds']},flush=True)


def frozen(p,args):
    fit=read(ROOT/p['raw_root']/'fit/training.json')
    if not fit['development_gate_passed'] or fit['protocol_sha256']!=sha(args.protocol):raise ValueError('Qualified frozen fit required.')
    for name,digest in fit['source_sha256'].items():
        if sha(ROOT/name)!=digest:raise ValueError('Frozen source/input changed: '+name)
    checkpoint=ROOT/fit['selected']['checkpoint']
    if sha(checkpoint)!=fit['selected']['checkpoint_sha256']:raise ValueError('Selected weights changed.')
    return fit,checkpoint


def export_or_evaluate(p,args):
    import openvino as ov
    exporting=args.mode=='export';raw=ROOT/p['raw_root']
    output=raw/('openvino/parity.json' if exporting else 'evaluation/evaluation.json')
    if output.exists() or (exporting and output.parent.exists()):raise FileExistsError('Preserve exports/evaluations.')
    fit,checkpoint=frozen(p,args)
    preflight=space(p,output,32*1024**2)
    torch.set_num_threads(2)
    network=MugDifferentialMotorNet().eval();network.load_state_dict(load_file(str(checkpoint)))
    if exporting:
        sample=(torch.zeros(1,5),torch.zeros(1,3))
        converted=ov.convert_model(torch.jit.trace(network,sample),example_input=sample)
        output.parent.mkdir();ov.save_model(converted,output.parent/'motor.xml',compress_to_fp16=False)
    else:
        parity=read(raw/'openvino/parity.json')
        if not parity['passed'] or parity['checkpoint_sha256']!=sha(checkpoint):raise ValueError('Passing export required.')
        for name,digest in parity['ir_sha256'].items():
            if sha(raw/'openvino'/name)!=digest:raise ValueError('Evaluated IR changed.')
    manifest,data=checked(p,args,'development' if exporting else 'evaluation')
    core=ov.Core();compiled=core.compile_model(str(raw/'openvino/motor.xml'),'CPU',{'PERFORMANCE_HINT':'LATENCY','INFERENCE_PRECISION_HINT':'f32','INFERENCE_NUM_THREADS':2})
    actual,latency=[],[]
    for q,d in zip(data['joints'],data['requested_translation']):
        start=time.perf_counter();actual.append(compiled([q[None],d[None]])[0].copy()[0]);latency.append((time.perf_counter()-start)*1000)
    actual=np.asarray(actual);reference=predict_torch(network,data,'cpu')
    errors=[]
    for a,b,d in zip(actual,reference,data['requested_translation']):
        x=limited_delta(a,d,p['step_sizing']['maximum_joint_delta_rad'])[0]
        y=limited_delta(b,d,p['step_sizing']['maximum_joint_delta_rad'])[0]
        errors.append(float(np.max(abs(x-y))))
    result=score(p,data,actual,fit['support'])
    result.update(schema=p['schema'],protocol_sha256=sha(args.protocol),checkpoint_sha256=sha(checkpoint),
        ir_sha256={name:sha(raw/'openvino'/name) for name in ('motor.xml','motor.bin')},source_sha256=sha(Path(__file__)),
        maximum_joint_delta_parity_rad=max(errors),all_parity_rad=errors,median_inference_ms=float(np.median(latency)),all_inference_ms=latency,
        device=core.get_property('CPU','FULL_DEVICE_NAME'),precision=str(compiled.get_property('INFERENCE_PRECISION_HINT')),preflight=preflight,
        passed=bool(result['summary']['gate_passed'] and max(errors)<=p['export']['maximum_joint_delta_parity_rad']),new_physical_trials=0,
        scope='Joint-limited local motor geometry; no physical, live-camera correction or Intel execution claim.')
    write(p,output,result)
    print({k:result[k] for k in ('passed','summary','maximum_joint_delta_parity_rad','median_inference_ms','device')},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',type=Path,default=ROOT/'docs/robotics/experiments/mug-correction-motor-v2.json')
    parser.add_argument('--mode',choices=['prepare','train','export','evaluate'],required=True)
    parser.add_argument('--split',choices=['training','development','evaluation'])
    args=parser.parse_args();args.protocol=args.protocol.resolve();p=read(args.protocol)
    sources(p,args.protocol)
    if args.mode=='prepare':
        if not args.split:parser.error('--split is required')
        prepare(p,args)
    elif args.mode=='train':train(p,args)
    else:export_or_evaluate(p,args)
