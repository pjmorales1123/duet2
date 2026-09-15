"""Headless physical trials using exactly the controller used by the local server."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mujoco
import numpy as np

from simulation_lab.autonomy import LiftReturn
from simulation_lab.scene import HOME, build_scene


def trial(seed, arm="auto", practice=True, kind="lift_return", transfer_side=None):
    xml, layout = build_scene(seed, practice=practice, transfer_side=transfer_side)
    model, started = mujoco.MjModel.from_xml_string(xml), time.perf_counter()
    data = mujoco.MjData(model)
    targets = np.array(HOME*2)
    data.qpos[:12], data.ctrl[:] = targets, targets
    for _ in range(200):
        mujoco.mj_step(model, data)
    data.time = 0.
    mujoco.mj_forward(model, data)
    task = LiftReturn(model, data, layout)
    task.start(arm, tube_id="A2" if kind == "transfer" else None, kind=kind, destination_slot="B2" if kind == "transfer" else None)
    peak_grip_torque = 0.
    hold_samples = []
    for _ in range(16000):
        before = data.qpos.copy()
        task.update(targets)
        # The teacher may edit its scratch IK data, never the physical coordinates.
        if not np.array_equal(before, data.qpos):
            raise AssertionError("Controller changed physical qpos outside integration")
        if model.neq or np.any(data.xfrc_applied) or np.any(data.qfrc_applied):
            raise AssertionError("Unexpected artificial attachment or applied prop force")
        if not task.active:
            break
        data.ctrl[:] = task.apply_gripper_limit(targets)
        mujoco.mj_step(model, data)
        if task.side:
            peak_grip_torque = max(peak_grip_torque, abs(float(data.actuator_force[task.offset+5])))
        if task.stage == "hold":
            hold_samples.append(float(data.xpos[task.body, 2]-task.origin[2]))
    result = {"seed": seed, "requested_arm": arm, "practice": practice, "kind": kind, "transfer_side": transfer_side, "wall_time_s": round(time.perf_counter()-started, 3),
              "peak_gripper_torque_nm": round(peak_grip_torque, 5),
              "measured_hold_min_cm": round(min(hold_samples)*100, 3) if hold_samples else None,
              "task": task.snapshot()}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(10))+[42])
    parser.add_argument("--arm", choices=("auto", "left", "right"), default="auto")
    parser.add_argument("--random-scene", action="store_true")
    parser.add_argument("--transfer", choices=("left", "right"), help="Evaluate A2 -> B2 in the chosen transfer layout")
    parser.add_argument("--output", default=".run/autonomy-evaluation.json")
    args = parser.parse_args()
    results = []
    for seed in args.seeds:
        result = trial(seed, args.transfer or args.arm, not args.random_scene, kind="transfer" if args.transfer else "lift_return", transfer_side=args.transfer)
        results.append(result)
        task = result["task"]
        print(f"Seed {seed}: {task['status']} · {task['arm']} {task['tube_id']} · {task['metrics']}", flush=True)
        if task["status"] != "succeeded":
            print(task["message"], flush=True)
    output = {"tested_at": datetime.now(timezone.utc).isoformat(), "engine": mujoco.__version__,
              "successes": sum(r["task"]["status"] == "succeeded" for r in results), "trials": len(results), "results": results}
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2)+"\n", encoding="utf-8")
    print(f"Saved {output['successes']}/{output['trials']} successes to {path}")


if __name__ == "__main__":
    main()
