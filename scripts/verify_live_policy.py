"""Exercise the same learned task class used by the live engine, without browser timing."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import mujoco,numpy as np
from simulation_lab.learned_task import LearnedBottleTask,DEFAULT_CHECKPOINT
from simulation_lab.scene import HOME
from scripts.evaluate_dinner_scene import load
from simulation_lab.storage import require_space


def run(a):
    require_space(a.output,32*1024**2)
    if Path(a.output).exists():raise FileExistsError('Use a fresh report path; previous evaluations are preserved.')
    if a.episode:
        folder=Path(a.episode);layout=json.loads((folder/'manifest.json').read_text())['layout']
        m=mujoco.MjModel.from_xml_path(str(folder/'scene.xml'));d=mujoco.MjData(m)
        mujoco.mj_setState(m,d,np.load(folder/'initial-integration-state.npy',allow_pickle=False),mujoco.mjtState.mjSTATE_INTEGRATION)
        mujoco.mj_forward(m,d)
    else:
        m,d,layout=load(a.seed)
        for _ in range(200):mujoco.mj_step(m,d)
    task=LearnedBottleTask(m,d,layout,checkpoint=a.checkpoint);targets=np.array(HOME*2);ticks=0;diagnostics=[]
    if a.blank:
        observation=task._observation
        task._observation=lambda: {k: v*0 for k,v in observation().items()}
    try:
        while task.active and ticks<12500:
            if a.cancel_after is not None and ticks>=a.cancel_after:task.cancel(targets);break
            if getattr(a,'diagnostics',False) and ticks%40==0:
                previous=task.policy.previous
                diagnostics.append({'elapsed_s':float(d.time-task.started),
                    'guard_error_rad':None if previous is None else float(np.max(np.abs(d.qpos[:5]-previous[0,4,:5].cpu().numpy()))),
                    'internal_progress_endpoints':task.policy.progress,'finger_support':task.metrics.get('both_fingers'),
                    'lift_cm':task.metrics.get('lift_cm'),'verified_hold_s':task.best_hold})
            before=d.qpos.copy();velocity=d.qvel.copy();task.update(targets)
            assert np.array_equal(before,d.qpos) and np.array_equal(velocity,d.qvel)
            assert m.neq==0 and not np.any(d.xfrc_applied) and not np.any(d.qfrc_applied)
            if task.active:d.ctrl[:]=task.apply_gripper_limit(targets);mujoco.mj_step(m,d)
            ticks+=1
        result=task.snapshot();result.update(input_case=Path(a.episode).name if a.episode else 'default-seed-'+str(a.seed),
                                            physical_state_writes=0,hidden_forces=0,equality_constraints=int(m.neq),
                                            renderer_closed=task.renderer is None)
        if getattr(a,'diagnostics',False):
            result['diagnostics']={'scope':'Read-only scoring diagnostics; never fed into policy actions. Guard error is sampled before each neural query.',
                                   'query_samples':diagnostics}
            if task.policy.context is not None and task.policy.meta.get('visual_encoder')=='bottle_rgb_geometry':
                feature=task.policy.context.detach().cpu().numpy()*task.policy.meta['visual_std']+task.policy.meta['visual_mean']
                result['diagnostics']['initial_rgb_centroid_pixels']=(feature[:2]*[320,240]).tolist()
        require_space(a.output,32*1024**2)
        Path(a.output).write_text(json.dumps(result,indent=2));print(json.dumps({'status':task.status,'message':task.message,'metrics':task.metrics}))
        return result
    finally:task.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--episode');p.add_argument('--seed',type=int,default=42);p.add_argument('--output',required=True)
    p.add_argument('--checkpoint',default=str(DEFAULT_CHECKPOINT))
    p.add_argument('--cancel-after',type=int);p.add_argument('--blank',action='store_true')
    p.add_argument('--diagnostics',action='store_true',help='Record read-only localization, tracking and support diagnostics.')
    result=run(p.parse_args());raise SystemExit(0 if result['status']=='succeeded' else 1)
