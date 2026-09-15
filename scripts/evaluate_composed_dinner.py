"""Physically execute a camera-grounded command through the browser's engine path."""
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
from simulation_lab.engine import LabEngine
from simulation_lab.storage import require_space

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frozen_inputs(protocol_path, suite_path):
    sources = sorted((ROOT / 'simulation_lab').glob('*.py')) + [Path(__file__).resolve()]
    suite = json.loads(suite_path.read_text(encoding='utf-8-sig'))
    model_files = {suite_path.relative_to(ROOT).as_posix(): sha(suite_path)}
    for relative in suite.values():
        folder = (suite_path.parent / relative).resolve()
        for path in sorted(folder.iterdir()):
            if path.is_file() and path.suffix in ('.safetensors', '.json', '.xml', '.bin'):
                model_files[path.relative_to(ROOT).as_posix()] = sha(path)
    return {'protocol_sha256': sha(protocol_path),
            'source_sha256': {p.relative_to(ROOT).as_posix(): sha(p) for p in sources},
            'model_files_sha256': model_files}


def run(args):
    protocol = json.loads(args.protocol.read_text())
    if args.seed not in protocol[args.split + '_seeds']:
        raise ValueError('Seed is outside the declared split.')
    frozen = frozen_inputs(args.protocol.resolve(), args.suite.resolve())
    if args.split == 'evaluation':
        if not args.freeze:
            raise ValueError('Final evaluation requires the development selection freeze.')
        selection = json.loads(args.freeze.read_text())
        if any(selection[k] != frozen[k] for k in frozen):
            raise ValueError('Current inputs differ from the final freeze.')
        if selection['evaluation_seeds'] != protocol['evaluation_seeds']:
            raise ValueError('Final seed list differs from the freeze.')
    if args.output.exists():
        raise FileExistsError(args.output)
    used = sum(p.stat().st_size for d in (ROOT / '.run').glob('composed-dinner-v1*')
               if d.is_dir() for p in d.rglob('*') if p.is_file())
    if used + 12 * 1024**2 > protocol['budget']['maximum_raw_and_packaged_mib'] * 1024**2:
        raise ValueError('Declared compact evidence budget would be exceeded.')
    require_space(args.output, 12 * 1024**2)
    args.output.mkdir(parents=True)
    for name in frozen['source_sha256']:
        destination = args.output / 'evaluated-source' / name
        require_space(destination, (ROOT / name).stat().st_size + 1024**2)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT / name).read_bytes())
    (args.output / 'frozen-inputs.json').write_bytes((json.dumps(frozen, indent=2) + '\n').encode())
    torch.set_num_threads(2)
    engine = LabEngine(width=320, height=240)
    engine.dinner_suite = args.suite
    trace = {k: [] for k in ('qpos', 'qvel', 'time', 'stage', 'targets')}
    started = time.perf_counter()
    report = {'seed': args.seed, 'split': args.split, 'instruction': protocol['instruction'],
              'bottle_start': protocol['bottle_start'], 'physics_state_writes_during_control': 0,
              'hidden_forces': 0, 'expected_steps': protocol['expected_steps'], 'frozen_inputs': frozen}
    try:
        engine._reset(seed=args.seed, scenario='dinner', dinner_preset='task',
                      drawer_open=False, bottle_start=protocol['bottle_start'])
        scene = ET.fromstring(engine.xml)
        compiler = scene.find('compiler')
        meshdir = Path(compiler.get('meshdir')).resolve()
        if not meshdir.is_relative_to(ROOT / 'simulation_lab/assets'):
            raise ValueError('Unexpected scene asset path.')
        compiler.set('meshdir', os.path.relpath(meshdir, args.output).replace('\\', '/'))
        (args.output / 'scene.xml').write_bytes(ET.tostring(scene, encoding='utf-8'))
        # This is the browser's real parser -> RGB scene observation -> planner
        # -> sequence construction path, executed in an isolated engine.
        engine._language_command({'text': protocol['instruction'], 'mode': 'learned_dinner'})
        report['visual_plan'] = engine.task.plan
        if engine.task.steps != protocol['expected_steps']:
            raise ValueError('RGB-grounded plan did not select the declared composed workflow.')
        limit = int(protocol['budget']['maximum_simulation_seconds_per_trial'] / engine.model.opt.timestep)
        for tick in range(limit):
            before, velocity = engine.data.qpos.copy(), engine.data.qvel.copy()
            engine.task.update(engine.target)
            if not np.array_equal(before, engine.data.qpos) or not np.array_equal(velocity, engine.data.qvel):
                raise RuntimeError('Controller wrote physical state.')
            if engine.model.neq or np.any(engine.data.xfrc_applied) or np.any(engine.data.qfrc_applied):
                raise RuntimeError('Unexpected constraint or applied force.')
            if tick % 10 == 0 or not engine.task.active:
                values = (engine.data.qpos.copy(), engine.data.qvel.copy(), float(engine.data.time),
                          engine.task.child.skill, engine.target.copy())
                for key, value in zip(trace, values):
                    trace[key].append(value)
            if tick % 2000 == 0:
                require_space(args.output, 12 * 1024**2)
            if not engine.task.active:
                break
            engine.data.ctrl[:] = engine.task.apply_gripper_limit(engine.target)
            mujoco.mj_step(engine.model, engine.data)
        if engine.task.active:
            engine.task.cancel(engine.target)
        report.update(status=engine.task.status, task=engine.task.snapshot())
    except ValueError as exc:
        report.update(status='refused', message=str(exc))
    finally:
        report.update(wall_seconds=time.perf_counter()-started,
                      simulation_seconds=float(engine.data.time) if hasattr(engine, 'data') else 0.)
        require_space(args.output, 12 * 1024**2)
        np.savez_compressed(args.output / 'states.npz', **{k: np.asarray(v) for k, v in trace.items()})
        (args.output / 'report.json').write_bytes((json.dumps(report, indent=2) + '\n').encode())
        if hasattr(engine, 'task') and hasattr(engine.task, 'close'):
            engine.task.close()
        engine.close()
    print(json.dumps({'seed': args.seed, 'status': report['status'],
                      'simulation_seconds': report['simulation_seconds'], 'wall_seconds': report['wall_seconds'],
                      'steps': report.get('visual_plan', {}).get('steps'),
                      'completed_steps': report.get('task', {}).get('completed_steps'),
                      'message': report.get('message', report.get('task', {}).get('message'))}), flush=True)
    return int(report['status'] != 'succeeded')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--split', choices=('development', 'evaluation'), required=True)
    parser.add_argument('--protocol', type=Path, default=Path('docs/robotics/experiments/composed-dinner-relay-v1.json'))
    parser.add_argument('--suite', type=Path, default=Path('models/dinner_suite/suite.json'))
    parser.add_argument('--freeze', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    raise SystemExit(run(parser.parse_args()))
