"""Inspect scene stability and candidate reach. This does not grasp objects."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mujoco
import numpy as np

from simulation_lab.autonomy import ArmIK, OPEN, PlanningError
from simulation_lab.dinner import dinner_state
from simulation_lab.scene import HOME, build_scene


def load(seed=42, preset="task", opened=False):
    xml, layout = build_scene(seed=seed, scenario="dinner", dinner_preset=preset, drawer_open=False)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    data.qpos[:12] = HOME*2
    data.ctrl[:] = HOME*2
    mujoco.mj_forward(model, data)
    return model, data, layout


def stability(model, data, layout, seconds=5.):
    # Deep contacts at initialization catch intersecting assets before settling.
    initial_penetration = max((max(0., -c.dist) for c in data.contact), default=0.)
    for _ in range(round(seconds/model.opt.timestep)):
        mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)
    records = []
    for item in layout["objects"]:
        body = data.body(item["body"])
        joint = model.joint(item["body"]+"_free")
        dof = joint.dofadr[0]
        records.append({"id": item["id"], "tilt_deg": float(np.rad2deg(np.arccos(np.clip(body.xmat[8], -1, 1)))),
                        "position_m": body.xpos.tolist(), "speed_m_s": float(np.linalg.norm(data.qvel[dof:dof+3])),
                        "angular_speed_rad_s": float(np.linalg.norm(data.qvel[dof+3:dof+6]))})
    passed = (np.isfinite(data.qpos).all() and initial_penetration < .0005 and
              all(r["tilt_deg"] < 5 and r["position_m"][2] > layout["table_z"]-.003 and
                  r["speed_m_s"] < .003 and r["angular_speed_rad_s"] < .15 for r in records))
    return {"seed": layout["seed"], "preset": layout["dinner_preset"], "source_zone": layout["source_zone"],
            "passed": bool(passed), "seconds": seconds, "initial_penetration_mm": initial_penetration*1000,
            "objects": records}


def candidate_reach(model, data, layout):
    results = []
    for item in layout["objects"]:
        for side in ("left", "right"):
            ik = ArmIK(model, data, side)
            point = data.site(item["grasp_site"]).xpos.copy()
            points = {}
            for stage, dz in (("grasp", 0.), ("above", .045), ("lift", .05)):
                try:
                    q = ik.solve(point+[0, 0, dz], np.array(HOME[:5]))
                    scratch = ik.data
                    scratch.qpos[ik.offset:ik.offset+5] = q
                    scratch.qpos[ik.offset+5] = OPEN
                    mujoco.mj_forward(model, scratch)
                    collisions = []
                    for c in scratch.contact:
                        a, b = int(c.geom1), int(c.geom2)
                        bodies = [model.body(model.geom_bodyid[g]).name for g in (a, b)]
                        if c.dist < -.0002 and any(n.startswith(side+"_") for n in bodies):
                            collisions.append([model.geom(g).name or bodies[i] for i, g in enumerate((a, b))])
                    points[stage] = {"ik": True, "position_error_mm": float(np.linalg.norm(ik.point(scratch)-(point+[0, 0, dz]))*1000),
                                     "blocking_contacts": collisions}
                except PlanningError:
                    points[stage] = {"ik": False}
            results.append({"object": item["id"], "arm": side, "points": points})
    return {"method": "Existing upright-finger IK at three isolated poses; not a trajectory, aperture test, grasp, or success result.", "candidates": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(".run/dinner-scene-evaluation.json"))
    args = parser.parse_args()
    trials = []
    for seed in [*range(10), 42]:
        for preset in ("task", "reference"):
            m, d, layout = load(seed, preset)
            trials.append(stability(m, d, layout))
    m, d, layout = load(42, "task", True)
    stability(m, d, layout)
    result = {"schema": "duet2.scene-audit.v1", "mujoco": mujoco.__version__,
              "passed": sum(t["passed"] for t in trials), "total": len(trials),
              "scope": "Scene initialization and settling only, not challenge task success. Dimensions are not randomized yet.",
              "trials": trials, "reach": candidate_reach(m, d, layout)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(f"Scene stability: {result['passed']}/{result['total']}. Saved {args.output}")
    for t in trials:
        if not t["passed"]:
            print("FAILED", t["seed"], t["preset"], "penetration", t["initial_penetration_mm"])
            print(t["objects"])
    for c in result["reach"]["candidates"]:
        if all(p.get("ik") and not p.get("blocking_contacts") for p in c["points"].values()):
            print("Clear isolated candidate poses:", c["object"], c["arm"])
    return int(result["passed"] != result["total"])


if __name__ == "__main__":
    raise SystemExit(main())
