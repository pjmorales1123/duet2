"""Does the live SmolVLA checkpoint actually do the task? Graded, not pass/fail.

Everything else in this repo measures the *speed* of the live policy
(benchmark_live_vla.py) or the numerical parity of its runtime
(export_smolvla_vision.py). Neither answers whether the policy accomplishes
anything, and `SmolVLATask` cannot answer it either: it reports 'succeeded' when
its timer expires, so a policy that never moved would still report success.

This script measures physical outcome instead, against the same declared final
poses the scripted teacher is graded on (`layout['targets']`):

  * did the object move at all,
  * did it move *toward* its declared target (progress ratio),
  * did it clear the teacher's strict placement gate (xy<6mm, z<3mm),
  * did the arms actually move, or did the policy sit still,
  * did it disturb objects it was not asked to touch.

Progress ratio is the honest headline for a checkpoint that may be far from
succeeding: 1.0 means the object reached the target, 0.0 means no net progress,
negative means it moved away. A strict pass/fail would report 'failed' for every
run and tell you nothing about whether the policy is learning the right thing.

Seeds come from the **validation** partition (2000-2019) and are enforced by
duet_protocol.require_partition: this is candidate assessment, which may select
a model but may not update weights. The evaluation partition (3000-3009) stays
untouched until a candidate is frozen.

    python scripts/evaluate_live_vla.py --seeds 2000 2001 --skill mug
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import mujoco
import numpy as np

INSTRUCTIONS = {'bottle': 'place the bottle on the table', 'plate': 'place the plate on the table',
                'mug': 'place the mug on the table', 'fork': 'place the fork on the table',
                'spoon': 'place the spoon on the table'}
# The teacher's own verify gate (dinner_autonomy.py), quoted so the two
# controllers are graded against one standard rather than two.
TEACHER_XY_M, TEACHER_Z_M = .006, .003


def trial(seed, skill, duration_s, mode='vla', settle_ticks=200):
    """One graded run. `mode` selects what drives the arms:

      'hold'    - nothing does. Arms stay at HOME. This is the FLOOR, and it is
                  not optional: objects start close to their declared targets,
                  so a do-nothing run can sit inside the teacher's 6mm gate
                  before any policy acts. Without this control, 'gate passed'
                  is unreadable.
      'vla'     - the live SmolVLA checkpoint.
      'teacher' - the scripted exact-state controller. This is the CEILING, and
                  it says how much of the remaining error is even achievable.

    All three are measured identically so the three numbers are comparable.
    """
    from simulation_lab.scene import HOME, build_scene
    from simulation_lab.vla_task import SmolVLATask, _CACHE

    xml, layout = build_scene(seed, practice=False, scenario='dinner', dinner_preset='task')
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    for _ in range(settle_ticks):  # let the seeded start come to rest first
        mujoco.mj_step(model, data)
    data.time = 0.

    item = next(o for o in layout['objects'] if o['id'] == skill)
    body = model.body(item['body']).id
    target = next(t for t in layout['targets'] if t['object_id'] == skill)
    goal = np.array(target['position_m'], dtype=float)
    others = {o['id']: model.body(o['body']).id for o in layout['objects'] if o['id'] != skill}

    start = data.xpos[body].copy()
    others_start = {name: data.xpos[index].copy() for name, index in others.items()}
    start_distance = float(np.linalg.norm(start[:2]-goal[:2]))

    if mode == 'teacher':
        from simulation_lab.dinner_autonomy import DinnerSequence
        task = DinnerSequence(model, data, layout)
        task.start(kind='dinner_place', object_id=skill)
    elif mode == 'vla':
        task = SmolVLATask(model, data, layout)
        task.start(INSTRUCTIONS[skill], duration_s=duration_s)
    else:
        task = None

    targets = np.array(HOME*2)
    previous = data.qpos[:12].copy()
    travel, ticks, started = 0., 0, time.perf_counter()
    limit = int(duration_s/model.opt.timestep) if task is None else 200000
    while ticks < limit and (task is None or task.active):
        before, velocity = data.qpos.copy(), data.qvel.copy()
        if task is not None:
            task.update(targets)
            assert np.array_equal(before, data.qpos), 'Controller changed authoritative positions'
            assert np.array_equal(velocity, data.qvel), 'Controller changed authoritative velocities'
            assert not np.any(data.xfrc_applied) and not np.any(data.qfrc_applied), 'Hidden applied force'
            if not task.active:
                break
            data.ctrl[:] = task.apply_gripper_limit(targets)
        else:
            data.ctrl[:] = targets  # 'hold': commanded to HOME, never updated
        mujoco.mj_step(model, data)
        travel += float(np.abs(data.qpos[:12]-previous).sum())
        previous = data.qpos[:12].copy()
        ticks += 1
    wall = time.perf_counter()-started

    end = data.xpos[body].copy()
    final_distance = float(np.linalg.norm(end[:2]-goal[:2]))
    z_error = abs(float(end[2]-goal[2]))
    disturbed = max((float(np.linalg.norm(data.xpos[index]-others_start[name]))
                     for name, index in others.items()), default=0.)
    inference = getattr(task, 'inference_ms', None) if task is not None else None
    if getattr(task, 'close', None):
        task.close()
    return {'seed': seed, 'skill': skill, 'mode': mode, 'instruction': INSTRUCTIONS[skill],
            'neural_runtime': _CACHE.get('runtime', 'PyTorch') if mode == 'vla' else None,
            'object_start_m': start.tolist(), 'object_end_m': end.tolist(), 'target_m': goal.tolist(),
            'object_displacement_m': float(np.linalg.norm(end-start)),
            'start_distance_to_target_m': start_distance, 'final_distance_to_target_m': final_distance,
            # 1.0 = reached the target, 0.0 = no net progress, negative = moved away.
            'progress_ratio': float((start_distance-final_distance)/start_distance) if start_distance > 1e-9 else 0.,
            'moved_toward_target': final_distance < start_distance,
            'placement_xy_error_mm': final_distance*1000, 'placement_z_error_mm': z_error*1000,
            'teacher_gate_passed': bool(final_distance < TEACHER_XY_M and z_error < TEACHER_Z_M),
            'other_object_max_displacement_mm': disturbed*1000,
            'arm_joint_travel_rad': travel, 'arms_moved': travel > .05,
            'simulation_seconds': float(data.time), 'wall_seconds': wall,
            'real_time_factor': float(data.time)/wall if wall > 0 else 0.,
            'inference_median_ms': float(np.median(inference)) if inference else None,
            'reported_status': getattr(task, 'status', 'held')}


def main():
    from simulation_lab.duet_protocol import require_partition
    from simulation_lab.vla_task import CHECKPOINT

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', type=int, nargs='+', default=[2000, 2001])
    parser.add_argument('--skill', choices=sorted(INSTRUCTIONS), nargs='+', default=['mug'])
    parser.add_argument('--modes', nargs='+', default=['hold', 'vla'], choices=('hold', 'vla', 'teacher'),
                        help="'hold' is the do-nothing floor and should almost always be included.")
    parser.add_argument('--duration-s', type=float, default=25.)
    parser.add_argument('--partition', default='validation',
                        help="Declared seed partition. 'evaluation' is held out; do not use it casually.")
    parser.add_argument('--output', type=Path, default=ROOT/'.run/live-vla-evaluation.json')
    args = parser.parse_args()

    seeds = require_partition(args.seeds, args.partition)
    trials = []
    for seed in seeds:
        for skill in args.skill:
            for mode in args.modes:
                result = trial(seed, skill, args.duration_s, mode)
                trials.append(result)
                print(json.dumps({k: result[k] for k in ('seed', 'skill', 'mode',
                    'start_distance_to_target_m', 'placement_xy_error_mm', 'progress_ratio',
                    'object_displacement_m', 'arms_moved', 'teacher_gate_passed')}), flush=True)

    def summarize(rows):
        if not rows:
            return None
        return {'trials': len(rows),
                'teacher_gate_passed': sum(r['teacher_gate_passed'] for r in rows),
                'moved_toward_target': sum(r['moved_toward_target'] for r in rows),
                'arms_moved': sum(r['arms_moved'] for r in rows),
                'median_progress_ratio': float(np.median([r['progress_ratio'] for r in rows])),
                'median_placement_xy_error_mm': float(np.median([r['placement_xy_error_mm'] for r in rows])),
                'median_object_displacement_mm': float(np.median([r['object_displacement_m'] for r in rows]))*1000}

    by_mode = {mode: summarize([t for t in trials if t['mode'] == mode]) for mode in args.modes}
    report = {'schema': 'duet-2.live-vla-evaluation.v2', 'checkpoint': CHECKPOINT.name,
        'partition': args.partition, 'seeds': list(seeds), 'skills': args.skill, 'modes': args.modes,
        'trials': len(trials), 'mujoco': mujoco.__version__,
        'median_start_distance_mm': float(np.median([t['start_distance_to_target_m'] for t in trials]))*1000,
        'by_mode': by_mode,
        'scope': 'Live SmolVLA capability probe, graded by physical outcome against the same declared final '
                 'poses the scripted teacher is graded on. Read by_mode comparatively: objects start close '
                 "to their targets, so the 'hold' (do-nothing) row is the floor and any claim about the "
                 "policy must beat it. 'teacher' is the achievable ceiling. Not a video submission and not "
                 'an Intel performance claim.',
        'rows': trials}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'checkpoint': report['checkpoint'], 'seeds': report['seeds'],
        'median_start_distance_mm': report['median_start_distance_mm'],
        'by_mode': report['by_mode']}, indent=2), flush=True)


if __name__ == '__main__':
    raise SystemExit(main())
