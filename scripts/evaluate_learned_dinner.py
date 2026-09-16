"""Physically evaluate neural dinner skills without teacher-generated actions."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import xml.etree.ElementTree as ET
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mujoco
import numpy as np
import torch
from simulation_lab.learned_dinner import LearnedDinnerTask, LearnedDinnerSequence
from simulation_lab.scene import build_scene, HOME
from simulation_lab.storage import require_space
from scripts.prepare_bottle_data import STATE


def run(args):
    if args.output.exists(): raise FileExistsError(args.output)
    require_space(args.output, 64*1024**2)
    capture=getattr(args,'capture',None)
    if capture:
        if capture.exists():raise FileExistsError(capture)
        require_space(capture,64*1024**2)
        capture.mkdir(parents=True)
    torch.set_num_threads(2)
    force=getattr(args,'disturbance_x_n',0.)
    duration=getattr(args,'disturbance_duration_s',.05)
    if not np.isfinite(force) or abs(force)>2 or (force and args.skills!='bottle'):
        raise ValueError('Declared disturbance supports a single bottle skill and force within +/-2 N.')
    if not np.isfinite(duration) or not .005<=duration<=.1:raise ValueError('Disturbance duration must be 5–100 ms.')
    disturbance_steps=0
    if args.episode:
        scene_path=args.episode.parent/'scene.xml'
        xml=scene_path.read_text(encoding='utf-8')
        model = mujoco.MjModel.from_xml_path(str(scene_path))
        data = mujoco.MjData(model)
        mujoco.mj_setState(model, data, np.load(args.episode/'initial-integration-state.npy', allow_pickle=False), STATE)
        layout = json.loads((args.episode/'manifest.json').read_text())['layout']
    else:
        xml, layout = build_scene(seed=args.seed, scenario='dinner', dinner_preset='task')
        model = mujoco.MjModel.from_xml_string(xml); data = mujoco.MjData(model)
        data.qpos[:12] = HOME*2; data.ctrl[:] = HOME*2
        if getattr(args,'bottle_start','upright')=='wide_left':
            from simulation_lab.dinner import left_reach_bottle_pose
            pose=left_reach_bottle_pose(args.seed);address=model.joint('bottle_free').qposadr[0]
            data.qpos[address:address+7]=[pose['x'],pose['y'],layout['table_z']+.001,np.cos(pose['yaw']/2),0,0,np.sin(pose['yaw']/2)]
        mujoco.mj_forward(model, data)
        for _ in range(300 if getattr(args,'bottle_start','upright')=='wide_left' else 200): mujoco.mj_step(model, data)
        data.time = 0.
    mujoco.mj_forward(model, data)
    if capture:
        scene=ET.fromstring(xml);compiler=scene.find('compiler')
        meshdir=Path(compiler.get('meshdir'))
        if not meshdir.is_absolute():meshdir=(args.episode.parent/meshdir).resolve()
        compiler.set('meshdir',os.path.relpath(meshdir,capture).replace('\\','/'))
        (capture/'scene.xml').write_text(ET.tostring(scene,encoding='unicode'),encoding='utf-8')
        trace={k:[] for k in ['qpos','qvel','time','stage','targets','progress']}
    if args.suite:
        paths = json.loads(args.suite.read_text(encoding='utf-8-sig'))
        checkpoints = {s: args.suite.parent/p for s, p in paths.items()}
        task = LearnedDinnerSequence(model, data, layout, checkpoints, args.skills.split(','))
    else:
        task = LearnedDinnerTask(model, data, layout, args.checkpoint, args.skills)
    targets = np.array(HOME*2)
    start = time.perf_counter()
    try:
        for tick in range(150000):
            before, velocity = data.qpos.copy(), data.qvel.copy()
            task.update(targets)
            assert np.array_equal(before, data.qpos) and np.array_equal(velocity, data.qvel)
            assert model.neq == 0 and not np.any(data.xfrc_applied) and not np.any(data.qfrc_applied)
            if capture and (tick%10==0 or not task.active):
                if tick%1000==0:require_space(capture,64*1024**2)
                child=task.child if hasattr(task,'child') else task
                for k,v in [('qpos',data.qpos.copy()),('qvel',data.qvel.copy()),('time',float(data.time)),
                            ('stage',child.skill),('targets',targets.copy()),('progress',child.policy.progress)]:trace[k].append(v)
            if not task.active: break
            data.ctrl[:] = task.apply_gripper_limit(targets)
            # Explicit test-harness intervention, outside the controller. Never
            # feed its magnitude, body pose or timing to the learned policy.
            if force and .8<=data.time<.8+duration:
                data.xfrc_applied[model.body('bottle').id,0]=force
                disturbance_steps+=1
            mujoco.mj_step(model, data)
            data.xfrc_applied[:]=0
        if task.active: task.cancel(targets)
        report = task.snapshot()
        report.update(seed=args.seed, episode=args.episode.name if args.episode else None,
                      bottle_start=getattr(args,'bottle_start','upright'),
                      wall_seconds=time.perf_counter()-start, physics_state_writes=0, hidden_forces=0,
                      equality_constraints=int(model.neq), evaluation='physical neural execution')
        model_paths=checkpoints if args.suite else {args.skills:args.checkpoint}
        report['checkpoint_sha256']={s:hashlib.sha256((p/'primitive.safetensors').read_bytes()).hexdigest() for s,p in model_paths.items()}
        if force:report['declared_disturbance']={'body':'bottle','force_x_n':force,'window_s':[.8,.8+duration],
                        'physics_steps':disturbance_steps,'source':'evaluation harness; not a policy input'}
        if capture:
            require_space(capture,64*1024**2)
            np.savez_compressed(capture/'states.npz',**{k:np.asarray(v) for k,v in trace.items()})
            (capture/'report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
        require_space(args.output, 64*1024**2)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8', newline='\n')
        print(json.dumps({'status': task.status, 'message': task.message, 'wall_seconds': report['wall_seconds'],
                          'metrics': report.get('metrics')}), flush=True)
        return int(task.status != 'succeeded')
    finally: task.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--suite', type=Path)
    p.add_argument('--episode', type=Path)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--bottle-start',choices=['upright','wide_left'],default='upright')
    p.add_argument('--skills', default='bottle')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--capture',type=Path,help='Preserve compact actual states for diagnosis/offline video; never policy inputs.')
    p.add_argument('--disturbance-x-n',type=float,default=0.,help='Explicit bottle-only test force for 0.8–0.85 simulated seconds.')
    p.add_argument('--disturbance-duration-s',type=float,default=.05,help='Duration of the declared test force; default 50 ms.')
    args = p.parse_args()
    if bool(args.checkpoint) == bool(args.suite): p.error('Choose checkpoint or suite.')
    raise SystemExit(run(args))
