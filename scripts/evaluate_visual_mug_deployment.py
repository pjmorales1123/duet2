"""Exposed seed-42 regressions of the real local and hosted mug entry points."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import patch
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import mujoco
import numpy as np
from PIL import Image
import torch
from simulation_lab.engine import LabEngine
from simulation_lab import mug_visual_profile, public_trial
from simulation_lab.storage import require_space
from scripts.package_spoon_release import physical_criteria


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def put(path, value):
    payload = (json.dumps(value, indent=2)+'\n').encode()
    require_space(path, len(payload)+1024)
    with path.open('xb') as stream:
        stream.write(payload)


def scene_copy(xml, output):
    tree = ET.fromstring(xml)
    tree.find('compiler').set('meshdir', os.path.relpath(ROOT/'simulation_lab/assets/so101/assets', output).replace('\\', '/'))
    payload = ET.tostring(tree, encoding='utf-8')
    require_space(output/'scene.xml', len(payload)+1024)
    with (output/'scene.xml').open('xb') as stream:
        stream.write(payload)


def run_case(args):
    output = args.output / (args.surface+'-'+args.preset)
    if output.exists():
        raise FileExistsError('Preserve every deployment attempt.')
    preflight = require_space(output, 64*1024**2)
    output.mkdir()
    torch.set_num_threads(2)
    profile, _ = mug_visual_profile.load_profile()
    declaration = json.loads((args.output/'protocol.json').read_text(encoding='utf-8-sig'))
    if any(sha(ROOT/name) != digest for name, digest in declaration['source_sha256'].items()):
        raise ValueError('A declared deployment source changed.')
    trace = {key: [] for key in ('qpos', 'qvel', 'time', 'stage', 'targets', 'progress')}
    def capture(model, data, task, targets):
        if len(trace['time']) % 200 == 0:
            require_space(output, 32*1024**2)
        for key, value in zip(trace, (data.qpos.copy(), data.qvel.copy(), float(data.time),
                task.child.skill, targets.copy(), task.child.policy.progress)):
            trace[key].append(value)
    start = time.perf_counter()
    result = {'surface': args.surface, 'preset': args.preset, 'seed': 42,
        'instruction': 'Set the table', 'profile_sha256': sha(mug_visual_profile.DEFAULT_PROFILE),
        'preflight': preflight, 'scope': 'Exposed deployment regression, not a new holdout.'}
    engine = None
    task_holder = []
    try:
        if args.surface == 'engine':
            engine = LabEngine(width=320, height=240)
            engine._reset(seed=42, scenario='dinner', dinner_preset='task', bottle_start=args.preset)
            scene_copy(engine.xml, output)
            engine._language_command({'text': 'Set the table', 'mode': 'learned_dinner_visual'})
            for tick in range(84000):
                before = (engine.data.qpos.copy(), engine.data.qvel.copy())
                engine.task.update(engine.target)
                if (not np.array_equal(before[0], engine.data.qpos) or not np.array_equal(before[1], engine.data.qvel)
                        or engine.model.neq or np.any(engine.data.xfrc_applied) or np.any(engine.data.qfrc_applied)):
                    raise RuntimeError('Physical-controller invariant failed.')
                if tick % 10 == 0 or not engine.task.active:
                    capture(engine.model, engine.data, engine.task, engine.target)
                if not engine.task.active:
                    break
                if time.perf_counter()-start > 420:
                    raise TimeoutError('Exposed deployment wall budget exceeded.')
                engine.data.ctrl[:] = engine.task.apply_gripper_limit(engine.target)
                mujoco.mj_step(engine.model, engine.data)
            if engine.task.active:
                engine.task.cancel(engine.target)
            snapshot = engine.task.snapshot()
        else:
            original_make = mug_visual_profile.make_sequence
            original_build = public_trial.build_scene
            original_step = mujoco.mj_step
            ticks = [0]
            def observe_make(*values, **kwargs):
                task = original_make(*values, **kwargs)
                task_holder.append(task)
                return task
            def observe_build(*values, **kwargs):
                xml, layout = original_build(*values, **kwargs)
                scene_copy(xml, output)
                return xml, layout
            def observe_step(model, data, *values, **kwargs):
                if task_holder:
                    if ticks[0] % 10 == 0:
                        capture(model, data, task_holder[0], data.ctrl)
                    ticks[0] += 1
                return original_step(model, data, *values, **kwargs)
            updates, final_image = [], None
            with patch.object(mug_visual_profile, 'make_sequence', observe_make), \
                    patch.object(public_trial, 'build_scene', observe_build), patch.object(mujoco, 'mj_step', observe_step):
                for rgb, message, report in public_trial.run_trial('Set the table', seed=42, controller='learned_visual',
                        bottle_start=args.preset, camera='opposite', cache_dir=output, max_wall_seconds=420):
                    updates.append({'message': message, 'report': report})
                    if rgb is not None:
                        final_image = rgb.copy()
                        if len(updates) == 1:
                            require_space(output/'initial.png', 1024**2)
                            with (output/'initial.png').open('xb') as stream:
                                Image.fromarray(rgb).save(stream, format='PNG')
                if not task_holder:
                    raise ValueError('The public entry point did not construct its verified task.')
                task = task_holder[0]
                capture(task.model, task.data, task, task.data.ctrl)
                snapshot = task.snapshot()
            put(output/'public-updates.json', updates)
            if final_image is not None:
                require_space(output/'final.png', 1024**2)
                with (output/'final.png').open('xb') as stream:
                    Image.fromarray(final_image).save(stream, format='PNG')
                if np.max(final_image) == 0:
                    raise ValueError('Hosted renderer returned a black final image.')
            result['public_report'] = updates[-1]['report']
        result['task'] = snapshot
        expected = (['bottle', 'plate', 'mug', 'drawer', 'fork', 'spoon'] if args.preset == 'upright'
            else ['reverse_bottle_right', 'reverse_bottle_left', 'plate', 'mug', 'drawer', 'fork', 'spoon'])
        if snapshot['status'] != 'succeeded' or snapshot['completed_steps'] != expected:
            raise ValueError('The full planned deployment workflow did not complete.')
        if snapshot.get('visual_feedback_profile') != profile['name']:
            raise ValueError('The promoted profile is not reported by the live task.')
        for row in snapshot['results']:
            physical_criteria(row)
        mug = next(row for row in snapshot['results'] if row['skill'] == 'mug')
        queries = mug.get('mug_visual_corrections', [])
        if mug.get('mug_correction_mode') != 'live' or not queries:
            raise ValueError('No live mug correction was observed.')
        result.update(passed=True, correction_queries=len(queries), physics_state_writes_during_control=0,
            hidden_forces=0, equality_constraints=0)
    except Exception as exc:
        result.update(passed=False, error_type=type(exc).__name__, message=str(exc))
        if engine is not None and hasattr(engine, 'task'):
            result.setdefault('task', engine.task.snapshot())
    finally:
        result.update(wall_seconds=time.perf_counter()-start, trace_frames=len(trace['time']))
        require_space(output/'states.npz', 32*1024**2)
        with (output/'states.npz').open('xb') as stream:
            np.savez_compressed(stream, **{key: np.asarray(value) for key, value in trace.items()})
        put(output/'report.json', result)
        if engine is not None:
            if hasattr(engine, 'task') and hasattr(engine.task, 'close'):
                engine.task.close()
            engine.close()
    print(json.dumps({key: result.get(key) for key in ('surface', 'preset', 'passed', 'wall_seconds', 'trace_frames', 'correction_queries', 'message')}), flush=True)
    return int(not result['passed'])


def run_all(args):
    if args.output.exists():
        raise FileExistsError('Use a new deployment regression folder.')
    preflight = require_space(args.output, 256*1024**2)
    args.output.mkdir(parents=True)
    sources = [Path(__file__), *(ROOT/'simulation_lab').glob('*.py'), ROOT/'hosting/gradio_app.py']
    put(args.output/'protocol.json', {'schema': 'talos.visual-mug-deployment.v1',
        'cases': [{'surface': surface, 'preset': preset, 'seed': 42} for surface in ('engine', 'public') for preset in ('upright', 'wide_left')],
        'instruction': 'Set the table', 'maximum_workers': 2, 'maximum_wall_seconds_per_trial': 420,
        'source_sha256': {p.relative_to(ROOT).as_posix(): sha(p) for p in sources},
        'profile_sha256': sha(mug_visual_profile.DEFAULT_PROFILE), 'preflight': preflight,
        'scope': 'All four exposed entry-point checks are declared before execution; no model selection or fresh holdout.'})
    def launch(case):
        surface, preset = case
        log = args.output/(surface+'-'+preset+'.log')
        if log.exists() or (args.output/(surface+'-'+preset)).exists():
            raise FileExistsError('Preserve case output and log.')
        require_space(log, 64*1024**2)
        with log.open('xb') as stream:
            code = subprocess.run([sys.executable, str(Path(__file__)), '--output', str(args.output),
                '--surface', surface, '--preset', preset], stdout=stream, stderr=subprocess.STDOUT).returncode
        report = json.loads((args.output/(surface+'-'+preset)/'report.json').read_text(encoding='utf-8-sig'))
        row = {'surface': surface, 'preset': preset, 'exit_code': code, 'passed': report['passed'],
            'report_sha256': sha(args.output/(surface+'-'+preset)/'report.json')}
        print(json.dumps(row), flush=True)
        return row
    with ThreadPoolExecutor(max_workers=2) as workers:
        results = list(workers.map(launch, [(s, p) for s in ('engine', 'public') for p in ('upright', 'wide_left')]))
    put(args.output/'batch.json', {'cases': results, 'passed': all(row['passed'] and row['exit_code'] == 0 for row in results)})
    return int(not all(row['passed'] and row['exit_code'] == 0 for row in results))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--surface', choices=['engine', 'public'])
    parser.add_argument('--preset', choices=['upright', 'wide_left'])
    args = parser.parse_args()
    raise SystemExit(run_case(args) if args.surface else run_all(args))
