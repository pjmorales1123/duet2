"""Compact six-skill physical demonstrations with independent endpoint replay.

Uses exact state only in the training teacher and physical scorer. Initial RGB,
2 Hz RGB observations, motor feedback and nominal 20 Hz actions are stored for
learning. Every episode retains failure evidence and is eligible only after both
teacher success and a contact-only replay with release and parked arms.
"""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mujoco
import numpy as np
from PIL import Image
from simulation_lab.dinner_autonomy import DinnerTask, SKILLS
from simulation_lab.dinner_monitor import DinnerPhysicalMonitor
from simulation_lab.policy_control import apply_targets
from simulation_lab.scene import HOME, build_scene
from simulation_lab.storage import require_space
from scripts.prepare_bottle_data import state, STATE


def save_json(path, value):
    require_space(path, 1024**2)
    path.write_text(json.dumps(value, indent=2)+'\n', encoding='utf-8', newline='\n')


# ponytail: hand-written paraphrases, not an LLM; enough lexical variety for a
# first VLA fine-tune. Expand/regenerate with a real paraphraser if time allows.
SKILL_INSTRUCTIONS = {
    'bottle':   ["Place the bottle on the table.", "Move the carafe down.", "Put the bottle in its spot."],
    'plate':    ["Place the dinner plate.", "Put the plate in its spot.", "Set the plate down."],
    'mug':      ["Place the mug above the plate.", "Put the cup down.", "Set the mug in its spot."],
    'fork':     ["Place the fork beside the plate.", "Put the fork down.", "Set the fork in its spot."],
    'spoon':    ["Place the spoon beside the plate.", "Put the spoon down.", "Set the spoon in its spot."],
}


def skill_instruction(skill, seed):
    options = SKILL_INSTRUCTIONS[skill]
    return options[seed % len(options)]


def render(renderer, data, camera='overview'):
    option = mujoco.MjvOption()
    option.geomgroup[3:] = 0
    renderer.update_scene(data, camera=camera, scene_option=option)
    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
    return renderer.render().copy()


def replay(model, initial, layout, skill, side, cap, endpoints, indices, folder):
    data = mujoco.MjData(model)
    mujoco.mj_setState(model, data, initial, STATE)
    mujoco.mj_forward(model, data)
    monitor = DinnerPhysicalMonitor(model, data, layout, skill, side)
    wrist_camera = 'left_wrist_cam' if side == 'left' else 'right_wrist_cam'
    frames, wrist_frames, joints, frame_indices = [], [], [], []
    renderer = mujoco.Renderer(model, height=240, width=320)
    try:
        for tick in range(int(indices[-1])+1+300):
            if tick % 100 == 0:
                require_space(folder, 64*1024**2)
                frames.append(render(renderer, data, 'overview'))
                wrist_frames.append(render(renderer, data, wrist_camera))
                joints.append(np.r_[data.qpos[:12], data.qvel[:12]])
                frame_indices.append(tick)
            if monitor.update(): break
            target = np.array([np.interp(tick, indices, endpoints[:, j]) for j in range(12)])
            data.ctrl[:] = apply_targets(model, data, target, cap, 0 if side == 'left' else 6)
            mujoco.mj_step(model, data)
            assert model.neq == 0 and not np.any(data.xfrc_applied) and not np.any(data.qfrc_applied)
        monitor.update()
    finally:
        renderer.close()
    return (monitor.report(), np.array(frames), np.array(wrist_frames),
            np.array(joints, dtype=np.float32), np.array(frame_indices))


def collect_episode(model, data, layout, skill, folder, side='auto'):
    require_space(folder, 96*1024**2)
    folder.mkdir()
    initial = state(model, data)
    np.save(folder/'initial-integration-state.npy', initial, allow_pickle=False)
    task = DinnerTask(model, data, layout)
    task.start(side=side, object_id=skill)
    targets = np.array(HOME*2)
    actions, stages = [], []
    cap = None
    for tick in range(20000):
        before, velocity = data.qpos.copy(), data.qvel.copy()
        task.update(targets)
        assert np.array_equal(before, data.qpos) and np.array_equal(velocity, data.qvel)
        if task.side:
            cap = .25 if skill == 'bottle' else task.grip_torque
            task.grip_torque = cap
            task.metrics['gripper_torque_limit_nm'] = cap
        if not task.active: break
        if tick % 200 == 0: require_space(folder, 96*1024**2)
        actions.append(targets.copy())
        stages.append(task.stage)
        data.ctrl[:] = task.apply_gripper_limit(targets)
        mujoco.mj_step(model, data)
        assert model.neq == 0 and not np.any(data.xfrc_applied) and not np.any(data.qfrc_applied)
    if task.active: task.cancel(targets)
    manifest = {'skill': skill, 'layout': deepcopy(layout), 'arm': task.side, 'gripper_cap_nm': cap,
                'outcome': task.snapshot(), 'training_eligible': False,
                'teacher': 'exact-state contact-only teacher', 'replay': None,
                'input_contract': 'Overhead RGB and motor feedback; teacher stages are offline training labels only',
                'state_writes_during_control': 0, 'external_forces': 0, 'equality_constraints': int(model.neq),
                'language': {'schema': 'duet-2.language.v1', 'instruction': skill_instruction(skill, layout['seed']),
                             'interpreter': 'hand_written_paraphrase', 'is_llm': False, 'is_vla': False}}
    if actions:
        raw = np.array(actions)
        # Preserve nominal full-rate actions even when compact replay fails.
        require_space(folder, 96*1024**2)
        np.savez_compressed(folder/'teacher-actions.npz', targets=raw, stages=np.array(stages))
    if task.status == 'succeeded':
        indices = np.unique(np.r_[np.arange(0, len(actions), 10), len(actions)-1])
        # Replay exactly the float32 values subsequently stored for training.
        endpoints = np.clip(raw[indices], model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1]).astype('float32')
        report, frames, wrist_frames, joints, frame_indices = replay(
            model, initial, layout, skill, task.side, cap, endpoints, indices, folder)
        manifest['replay'] = report
        manifest['replayed_saved_float32_endpoints'] = True
        manifest['training_eligible'] = report['passed']
        require_space(folder, 96*1024**2)
        np.savez_compressed(folder/'trajectory.npz', actions20=endpoints.astype('float32'),
                            action_indices=indices, stages=np.array(stages)[indices], physics_steps=len(actions))
        np.savez_compressed(folder/'observations.npz', overhead=frames, wrist=wrist_frames,
                             state=joints, action_indices=frame_indices)
        Image.fromarray(frames[0]).save(folder/'overhead.png')
        Image.fromarray(wrist_frames[0]).save(folder/'wrist.png')
    save_json(folder/'manifest.json', manifest)
    return {'skill': skill, 'teacher_status': task.status, 'message': task.message,
            'arm': task.side, 'training_eligible': manifest['training_eligible'], 'replay': manifest['replay']}


def collect_sequence(seed, out, dinner_variation="duet_v1", dinner_layout=None, skills=None):
    require_space(out, 600*1024**2)
    out.mkdir(parents=True)
    xml, layout = build_scene(seed=seed, scenario='dinner', dinner_preset='task',
                              dinner_variation=dinner_variation, dinner_layout=dinner_layout)
    model = mujoco.MjModel.from_xml_string(xml)
    model.vis.quality.offsamples = 0
    data = mujoco.MjData(model)
    data.qpos[:12] = HOME*2
    data.ctrl[:] = HOME*2
    mujoco.mj_forward(model, data)
    for _ in range(200): mujoco.mj_step(model, data)
    data.time = 0.
    scene = ET.fromstring(xml)
    compiler = scene.find('compiler')
    compiler.set('meshdir', os.path.relpath(compiler.get('meshdir'), out).replace('\\', '/'))
    (out/'scene.xml').write_text(ET.tostring(scene, encoding='unicode'), encoding='utf-8')
    rows = []
    planned_skills = tuple(skills or SKILLS)
    for skill in planned_skills:
        rows.append(collect_episode(model, data, layout, skill, out/skill))
        report = {'schema': 'duet2.dinner-learning-collection.v1', 'seed': seed, 'split': 'training',
                  'planned_skills': list(planned_skills), 'episodes': rows,
                  'eligible': sum(r['training_eligible'] for r in rows), 'simulation_seconds': float(data.time)}
        save_json(out/'summary.json', report)
        print(json.dumps({'seed': seed, **rows[-1]}), flush=True)
        if rows[-1]['teacher_status'] != 'succeeded': break
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seeds', default='42', help='Explicit training/development seeds, never a hidden holdout.')
    parser.add_argument('--protocol', type=Path, default=Path('docs/robotics/experiments/dinner-learning-v1.json'))
    args = parser.parse_args()
    seeds = [int(s) for s in args.seeds.split(',')]
    if not seeds or len(seeds) > 32 or len(set(seeds)) != len(seeds): raise ValueError('Use 1–32 unique seeds.')
    protocol=json.loads(args.protocol.read_text(encoding='utf-8'))
    if any(s not in protocol['training_seeds'] for s in seeds):
        raise ValueError('Collection requires declared training seeds; development/evaluation seeds cannot become demonstrations.')
    if args.output.exists(): raise FileExistsError('Existing batches are preserved; choose a fresh output.')
    require_space(args.output, len(seeds)*600*1024**2)
    args.output.mkdir(parents=True)
    save_json(args.output/'protocol.json', {'seeds': seeds, 'split': 'training', 'skills': list(SKILLS),
              'storage_reserve_gib': 10, 'purpose': 'Six-skill feasibility/training; no evaluation claim',
              'declared_protocol':args.protocol.as_posix()})
    rows = []
    for seed in seeds:
        rows.append(collect_sequence(seed, args.output/f'seed-{seed}'))
        save_json(args.output/'summary.json', {'sequences': rows})
    return int(any(r['eligible'] != len(SKILLS) for r in rows))


if __name__ == '__main__':
    raise SystemExit(main())
