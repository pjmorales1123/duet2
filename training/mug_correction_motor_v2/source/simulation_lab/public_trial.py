"""Isolated, bounded physical trials for the hosted Gradio demonstration.

Each request owns its scene and controllers. This module does not load local
credentials, the browser server, recordings, or any user's active simulation.
"""
import json
from pathlib import Path
import time

import mujoco
import numpy as np

from .language import CommandError, parse_command
from .scene import CAMERAS, HOME, build_scene
from .storage import require_space


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUITE = ROOT / 'models' / 'dinner_suite' / 'suite.json'


def load_checkpoints(suite=None):
    path = Path(suite) if suite else DEFAULT_SUITE
    if not suite and not path.is_file():
        path = ROOT / 'models' / 'bottle_relays' / 'suite.json'
    if not path.is_file():
        return {'bottle': ROOT / 'models' / 'bottle_visual'}
    paths = json.loads(path.read_text(encoding='utf-8-sig'))
    return {name: path.parent / relative for name, relative in paths.items()}


def validate_request(instruction, seed, controller, bottle_start, camera):
    plan = parse_command(instruction)
    if plan.get('control') == 'cancel':
        raise CommandError('Use Cancel trial to stop a running trial.')
    if isinstance(seed, bool) or not isinstance(seed, (int, float)) or not np.isfinite(seed) or int(seed) != seed or not 0 <= seed <= 2147483647:
        raise CommandError('Choose an integer seed between 0 and 2147483647.')
    if controller not in ('learned', 'programmed'):
        raise CommandError('Choose learned or programmed control.')
    if bottle_start not in ('upright', 'wide_left') or camera not in CAMERAS:
        raise CommandError('Choose one of the listed scene and camera options.')
    return plan, int(seed)


def trial_report(task, seed, controller, bottle_start, started):
    snapshot = task.snapshot()
    results = snapshot.get('results', [])
    plan = snapshot.get('intent_plan', {})
    if not isinstance(plan, dict):
        plan = {'steps': snapshot.get('steps', [])}
    return {
        'status': snapshot['status'], 'message': snapshot['message'],
        'seed': seed, 'controller': controller, 'bottle_start': bottle_start,
        'simulation_seconds': snapshot.get('elapsed_s', 0),
        'wall_seconds': round(time.perf_counter() - started, 2),
        'progress': snapshot.get('progress', 0),
        'completed_steps': snapshot.get('completed_steps', []),
        'results': [{k: r[k] for k in ('skill', 'status', 'message', 'metrics') if k in r} for r in results],
        'plan': plan.get('steps', snapshot.get('steps', [])),
        'camera_plan_reasons': plan.get('reasons', []),
        'physics_state_writes_during_control': 0, 'hidden_forces': 0,
        'control_inputs': 'Initial RGB views and motor feedback; separate physical stop/score monitor.' if controller == 'learned' else 'Programmed controller using exact simulator state.',
    }


def run_trial(instruction, seed=42, controller='learned', bottle_start='upright',
              camera='opposite', *, suite=None, cache_dir=None, max_wall_seconds=300, cancel_event=None, policy_factory=None):
    """Yield RGB, progress text, and a small public report; never write episodes."""
    renderer = task = None
    started = time.perf_counter()
    try:
        plan, seed = validate_request(instruction, seed, controller, bottle_start, camera)
        require_space(cache_dir or ROOT, 128 * 1024**2)
        xml, layout = build_scene(seed=seed, scenario='dinner', dinner_preset='task')
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        data.qpos[:12] = HOME * 2
        data.ctrl[:] = HOME * 2
        if bottle_start == 'wide_left':
            from .dinner import left_reach_bottle_pose
            pose = left_reach_bottle_pose(seed)
            address = model.joint('bottle_free').qposadr[0]
            data.qpos[address:address+7] = [pose['x'], pose['y'], layout['table_z']+.001,
                                         np.cos(pose['yaw']/2), 0, 0, np.sin(pose['yaw']/2)]
        mujoco.mj_forward(model, data)
        for _ in range(300 if bottle_start == 'wide_left' else 200):
            mujoco.mj_step(model, data)
        data.time = 0.
        model.vis.quality.offsamples = 0
        renderer = mujoco.Renderer(model, height=360, width=640)
        option = mujoco.MjvOption()
        option.geomgroup[3:] = 0

        def frame():
            renderer.update_scene(data, camera=camera, scene_option=option)
            renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
            return renderer.render().copy()

        yield frame(), 'Reading the instruction and preparing a fresh scene…', {'status': 'preparing', 'seed': seed}
        if controller == 'learned':
            from .learned_dinner import LearnedDinnerSequence, observe_scene
            from .learned_plan import plan_learned_steps
            checkpoints = load_checkpoints(suite)
            visual_plan = plan_learned_steps(plan, observe_scene(model, data), checkpoints)
            options = {'policy_factory': policy_factory} if policy_factory else {}
            task = LearnedDinnerSequence(model, data, layout, checkpoints, visual_plan['steps'], **options)
            task.plan = visual_plan
        else:
            from .command_task import CommandSequence
            task = CommandSequence(model, data, layout)
            task.start_plan(plan)
        targets = np.array(HOME * 2)
        last_yield = time.perf_counter()
        timed_out = False
        for tick in range(120000):
            if cancel_event is not None and cancel_event.is_set():
                task.cancel(targets)
                break
            before, velocity = data.qpos.copy(), data.qvel.copy()
            task.update(targets)
            if not np.array_equal(before, data.qpos) or not np.array_equal(velocity, data.qvel) or model.neq or np.any(data.xfrc_applied) or np.any(data.qfrc_applied):
                raise RuntimeError('Physical-controller invariant failed.')
            now = time.perf_counter()
            if not task.active:
                break
            if now - started > max_wall_seconds:
                timed_out = True
                task.cancel(targets)
                break
            data.ctrl[:] = task.apply_gripper_limit(targets)
            mujoco.mj_step(model, data)
            if tick % 200 == 0 and now - last_yield >= .5:
                require_space(cache_dir or ROOT, 128 * 1024**2)
                report = trial_report(task, seed, controller, bottle_start, started)
                yield frame(), f"{report['progress']:.0%} · {report['simulation_seconds']:.1f} simulated seconds · {report['message']}", report
                last_yield = time.perf_counter()
        if task.active:
            timed_out = True
            task.cancel(targets)
        report = trial_report(task, seed, controller, bottle_start, started)
        if timed_out:
            report['message'] = 'Trial stopped at the hosting time limit. Try a shorter instruction.'
        yield frame(), f"{report['status'].capitalize()} · {report['message']}", report
    except (CommandError, ValueError) as exc:
        yield None, str(exc), {'status': 'refused', 'message': str(exc)}
    finally:
        if task is not None and hasattr(task, 'close'):
            task.close()
        if renderer is not None:
            renderer.close()
