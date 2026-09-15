"""Fit a compact camera-conditioned neural trajectory, without copying images."""
import argparse,json,sys,time,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,torch
from safetensors.torch import save_file
from simulation_lab.primitive_policy import PrimitiveNet
from simulation_lab.storage import require_space,GIB

def main(a):
    if a.steps<=0 or a.save_every<=0:raise ValueError('Positive steps required.')
    torch.set_num_threads(2);torch.manual_seed(31)
    out=Path(a.output)
    if out.exists():raise FileExistsError(out)
    used=sum(f.stat().st_size for p in Path('.run').glob('bottle-sequence*') for f in (p.rglob('*') if p.is_dir() else [p]) if f.is_file())
    planned=((a.steps+a.save_every-1)//a.save_every)*3*1024**2
    if used+planned>2*GIB:raise OSError('Sequence-goal budget exceeded.')
    require_space(out,planned);out.mkdir()
    source=Path(a.source)
    with np.load(source/'retrieval.npz',allow_pickle=False) as z:d={k:z[k] for k in z.files}
    info=json.loads((source/'retrieval.json').read_text())
    if info.get('action_preprocessing'):
        gate=json.loads((source/'input-replay.json').read_text())
        if not gate['passed'] or gate['source_sha256']!=hashlib.sha256((source/'retrieval.npz').read_bytes()).hexdigest():
            raise ValueError('Modified action labels require passing physical replay for every fitted episode.')
    initial=d['visual'][d['bounds'][:,0]];vm=initial.mean(0);vs=np.maximum(initial.std(0),.01)
    if a.visual_scaling=='global':vs[:]=max(float(vs.max()),.01)
    visual=[];seconds=[]
    for start,end in d['bounds']:
        visual.append(np.repeat(((d['visual'][start]-vm)/vs)[None],end-start,axis=0));seconds.append(d['seconds'][start:end] if 'seconds' in d else np.arange(end-start)/20)
    am=d['actions'].mean(0);ast=np.maximum(d['actions'].std(0),.02)
    v,t,y=[torch.tensor(x,device='cuda',dtype=torch.float32) for x in [np.concatenate(visual),np.concatenate(seconds),(d['actions']-am)/ast]]
    net=PrimitiveNet().cuda()
    if a.warm_start:
        from safetensors.torch import load_file
        previous=json.loads((Path(a.warm_start)/'primitive.json').read_text())
        if previous['source_sha256']!=hashlib.sha256((source/'retrieval.npz').read_bytes()).hexdigest():raise ValueError('Training source changed.')
        net.load_state_dict(load_file(str(Path(a.warm_start)/'primitive.safetensors'),device='cuda'))
    opt=(torch.optim.LBFGS(net.parameters(),lr=a.lr,max_iter=10,history_size=10,line_search_fn='strong_wolfe',tolerance_change=a.lbfgs_tolerance_change,tolerance_grad=a.lbfgs_tolerance_grad)
         if a.optimizer=='lbfgs' else torch.optim.AdamW(net.parameters(),lr=a.lr,weight_decay=1e-6))
    meta={'training_episodes':info['episodes'],'source_sha256':hashlib.sha256((source/'retrieval.npz').read_bytes()).hexdigest(),
          'visual_mean':vm.tolist(),'visual_std':vs.tolist(),'action_mean':am.tolist(),'action_std':ast.tolist(),
          'max_seconds':float(t.max()),'reconstruction_limit':info['reconstruction_limit'],'tracking_tolerance_rad':.005,
          'visual_preprocess':info.get('visual_preprocess','raw'),'reconstruction_check':info.get('reconstruction_check','every_query'),
          'visual_encoder':info.get('visual_encoder','camera_pca'),
          'arm_offset':info.get('arm_offset',0),'skill':info.get('skill','bottle'),
          'gripper_cap_nm':info.get('gripper_cap_nm',.25),
          'cameras':info.get('cameras',['overhead','left_wrist_cam','right_wrist_cam']),
          'arguments':vars(a),'architecture':'Three 256-unit SiLU layers; initial 32-dimensional visual encoding plus 13 internal-progress Fourier features',
          'scope':'Learned motion primitive with measured-joint progress guard; no continuous visual correction.'}
    for name in ['logical_skill','trained_destination_m','feature_support','action_preprocessing']:
        if name in info:meta[name]=info[name]
    began=time.perf_counter()
    for step in range(1,a.steps+1):
        idx=torch.arange(len(t),device='cuda') if a.optimizer=='lbfgs' else torch.randint(len(t),(1024,),device='cuda')
        def closure():
            opt.zero_grad(set_to_none=True)
            value=torch.mean((net(v[idx],t[idx])-y[idx])**2)
            if not torch.isfinite(value):raise FloatingPointError('Nonfinite loss')
            value.backward()
            if a.optimizer!='lbfgs':torch.nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True)
            return value
        if a.optimizer=='lbfgs':loss=opt.step(closure)
        else:loss=closure();opt.step()
        if a.decay:
            for group in opt.param_groups:group['lr']=a.lr*(.05+.95*(1+np.cos(np.pi*step/a.steps))/2)
        if step%1000==0 or (a.optimizer=='lbfgs' and step%10==0):print(json.dumps({'step':step,'loss':float(loss.detach()),'elapsed_s':time.perf_counter()-began}),flush=True)
        if step%a.save_every==0 or step==a.steps:
            require_space(out,32*1024**2)
            folder=out/f'step-{step:06d}';folder.mkdir()
            save_file({k:x.detach().cpu().contiguous() for k,x in net.state_dict().items()},str(folder/'primitive.safetensors'))
            np.savez_compressed(folder/'visual.npz',**{k:d[k] for k in ['mean','components','scale']})
            (folder/'primitive.json').write_text(json.dumps(meta,indent=2))
            (folder/'talos_normalization.json').write_text(json.dumps({'observation.state':{'mean':[0.]*24,'std':[1.]*24},'action':{'mean':[0.]*12,'std':[1.]*12},'images':{'mean':[0.]*3,'std':[1.]*3}}))
            print('CHECKPOINT',folder,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--steps',type=int,default=40000)
    p.add_argument('--save-every',type=int,default=10000);p.add_argument('--lr',type=float,default=.0003)
    p.add_argument('--warm-start');p.add_argument('--decay',action='store_true')
    p.add_argument('--source',default='.run/bottle-robustness-model')
    p.add_argument('--visual-scaling',choices=['standard','global'],default='standard')
    p.add_argument('--optimizer',choices=['adam','lbfgs'],default='adam')
    p.add_argument('--lbfgs-tolerance-change',type=float,default=1e-12)
    p.add_argument('--lbfgs-tolerance-grad',type=float,default=1e-9);main(p.parse_args())
