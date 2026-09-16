"""Load the live SmolVLA checkpoint (simulation_lab/vla_task.py) against a real
dinner scene and report whether it can keep up with the physics loop.

This is the same code path simulation_lab.engine.LabEngine uses for a
`{"mode": "learned_vla"}` language command, run headless (no server, no
browser) so it can be timed directly. Requires the full requirements.txt
stack (torch, lerobot, transformers) installed, and a checkpoint present at
models/duet-smolvla-v1 (see the root README for how to fetch one).

Usage:
    python scripts/benchmark_live_vla.py --instruction "place the mug on the table"
    python scripts/benchmark_live_vla.py --seed 1000 --sim-seconds 5

See simulation_lab/PERF_NOTES.md for the current baseline numbers and the
optimizations already tried against this checkpoint.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import time
import mujoco
import numpy as np

from simulation_lab.scene import build_scene, HOME
from simulation_lab.vla_task import SmolVLATask

# engine.LabEngine._run() batches at most this much sim time per outer loop
# iteration (its accumulator is capped at .1s), and SmolVLATask.QUERY_EVERY_TICKS
# ticks at the scene's 0.005s timestep lands one inference call inside that
# same window - so this is the wall-clock budget one VLA query has to stay
# real-time before the sim visibly stutters (see engine.py, vla_task.py).
QUERY_BUDGET_MS = 100.0


def run(seed: int, instruction: str, sim_seconds: float):
    print(f"Building dinner scene (seed={seed})...")
    xml, layout = build_scene(seed, practice=False, scenario="dinner", dinner_preset="task")
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    target = np.asarray(HOME * 2, dtype=float)
    data.qpos[:12] = target
    data.ctrl[:] = target
    for _ in range(200):
        mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)

    print(f'Loading checkpoint and starting: "{instruction}"...')
    t0 = time.perf_counter()
    task = SmolVLATask(model, data, layout)
    task.start(instruction)
    cold_start_s = time.perf_counter() - t0
    print(f"cold start (checkpoint load + renderer setup): {cold_start_s:.1f}s")

    render_ms = [t for _ in range(6) for t in [_timed(lambda: task._frame("overview"))]]

    n_ticks = round(sim_seconds / model.opt.timestep)
    print(f"Running {n_ticks} physics ticks ({sim_seconds:.1f}s sim time, "
          f"~{n_ticks // task.QUERY_EVERY_TICKS} VLA queries)...")
    wall0 = time.perf_counter()
    for _ in range(n_ticks):
        task.update(target)
        data.ctrl[:12] = task.apply_gripper_limit(target)[:12]
        mujoco.mj_step(model, data)
    wall_s = time.perf_counter() - wall0

    inference_ms = task.inference_ms
    chunk = getattr(task.policy.config, "chunk_size", 1)
    task.close()

    # SmolVLA is an action-chunk policy: only every `chunk_size`-th query runs a
    # real forward pass, the rest pop a cached action. Reporting one blended
    # "inference" number hides a ~4-orders-of-magnitude gap between the two, so
    # split them - the expensive tail is what actually sets the real-time factor.
    refills = [t for t in inference_ms if t > 10 * np.median(inference_ms)]
    pops = [t for t in inference_ms if t not in refills]

    print()
    print("=== Results ===")
    print(f"render (per frame, n=6):  median={np.median(render_ms):.1f}ms  mean={np.mean(render_ms):.1f}ms")
    print(f"query, cached action (n={len(pops)}): median={np.median(pops):.1f}ms" if pops else "no cached-action queries")
    if refills:
        print(f"query, real forward pass (n={len(refills)}, 1 per {chunk} queries): "
              f"median={np.median(refills):.0f}ms  max={np.max(refills):.0f}ms")
    else:
        print(f"no chunk refill observed - run with a longer --sim-seconds "
              f"(need >{chunk * 20} ticks) to measure a real forward pass")
    sim_s = n_ticks * model.opt.timestep
    real_time_factor = sim_s / wall_s
    print(f"sim_time={sim_s:.2f}s wall_time={wall_s:.2f}s real_time_factor={real_time_factor:.2f}")
    # Amortised cost of one query: a forward pass every `chunk` queries, plus the
    # two camera frames that forward pass consumes (vla_task only renders then).
    amortised_ms = ((np.median(refills) if refills else 0.0) + 2 * np.median(render_ms)) / chunk + np.median(pops or [0])
    verdict = "OK" if amortised_ms < QUERY_BUDGET_MS else "LAGGING"
    print(f"amortised per-query cost: {amortised_ms:.1f}ms vs {QUERY_BUDGET_MS:.0f}ms budget -> {verdict}")
    return {"cold_start_s": cold_start_s, "render_ms": render_ms, "inference_ms": inference_ms,
            "refill_ms": refills, "pop_ms": pops,
            "real_time_factor": real_time_factor, "verdict": verdict}


def _timed(fn):
    t0 = time.perf_counter()
    fn()
    return (time.perf_counter() - t0) * 1000


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed", type=int, default=1000, help="Training-split seed for the dinner scene (default 1000).")
    p.add_argument("--instruction", default="place the mug on the table")
    p.add_argument("--sim-seconds", type=float, default=3.0,
                   help="Simulated seconds to run (default 3.0, ~6 VLA queries at the 10Hz query cadence).")
    args = p.parse_args()
    run(args.seed, args.instruction, args.sim_seconds)
