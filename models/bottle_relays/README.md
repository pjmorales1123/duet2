# Learned table-supported bottle relays, revision v3

Four new neural policies coordinate two sequential grasps. Each arm physically
releases the bottle onto the table and parks before the other arm starts.
The original `models/bottle_visual` and `models/bottle_primitive` remain unchanged.

| Direction | Trained start | Fixed final bottle position | Frozen evaluation |
| --- | --- | --- | --- |
| Left to right | Upright task-preset bottle region | x=0.20, y=0.05 m | 5/5 seeds, 1.13–1.69 mm XY error |
| Right to left | `wide_left`: x=-0.07341, y=-0.12157 m, each ±6 mm | x=0.10, y=-0.115 m | 5/5 seeds, 0.85–1.32 mm XY error |

Both use a shared intermediate position x=0, y=-0.10 m. Positions are measured
in the simulation's world frame. These are two bounded regions and fixed goals;
they do not establish arbitrary reachable-position coverage or airborne handoffs.

Inference uses a classical overhead RGB bottle centroid, a neural trajectory,
and measured-joint progress feedback. No teacher, simulator object pose, action
retrieval or offline stage label generates runtime motor targets. A separate
physical monitor uses privileged simulator state to stop/score contact, lift,
release, parking, placement and unintended collisions. The fixed destination in
model metadata is used by this monitor only. Continuous visual correction is
unfinished.

Each leg has 13 training examples. An offline repair sustains the closed finger
target during carry stages; all 52 aligned training trajectories passed physical
replay before fitting. The model architecture remains three 256-unit SiLU layers.
Training used 16,000 Adam steps followed by at most 400 full-batch L-BFGS steps on
the RTX 4070. Initial fits that lost grip are retained in the experiment archive.

The packaged OpenVINO FP32 models passed parity and all ten frozen physical
trials. CPU benchmark timings in these folders are from the **AMD Ryzen 7 5800X**;
they are not evidence of final Intel execution. The GPU entries identify the
RTX 4070, also not Intel hardware evidence.

Evidence: [all outcomes and freezes](../../docs/robotics/evidence/learned-relays-v3/summary.json).
Inputs and retraining: [training notes](../../training/bottle_relays/README.md).

```powershell
./start-lab.ps1 -Learned -DinnerSuite models/bottle_relays/suite.json
```

Choose the learned dinner controller. With a standard upright start, say/type
“pass the bottle to the right arm”. With the farther-left start, use “place the
bottle” or “pass the bottle to the left arm”. The suite also references the
preserved direct bottle-placement model. Other dinner models are not included
in this relay-only suite.

```powershell
./.venv-training/Scripts/python.exe scripts/evaluate_learned_dinner.py --suite models/bottle_relays/suite.json --skills relay_bottle_left,relay_bottle_right --seed 2026092501 --output .run/relay-reproduction-2501.json
./.venv-training/Scripts/python.exe scripts/evaluate_learned_dinner.py --suite models/bottle_relays/suite.json --skills reverse_bottle_right,reverse_bottle_left --bottle-start wide_left --seed 2026092601 --output .run/reverse-reproduction-2601.json
```

These are now exposed reproduction seeds. All new output paths must be unused;
each export retains the 10 GiB reserve. Original Talos code, data and these
weights use the root MIT license. Robot asset notices remain separate.
